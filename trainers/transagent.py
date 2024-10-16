import os.path as osp
from collections import OrderedDict
import math
import pickle
import json
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

import knowledge_extraction
from knowledge_extraction import load_pretrained_weights as load_experts, vit_base, sam_vit_base, vitdet_base

_tokenizer = _Tokenizer()

DS_PATH = {
    'Caltech101': 'caltech-101',
    'DescribableTextures': 'dtd',
    'EuroSAT': 'eurosat',
    'FGVCAircraft': 'fgvc_aircraft',
    'Food101': 'food-101',
    'ImageNet': 'imagenet',
    'OxfordFlowers': 'oxford_flowers',
    'OxfordPets': 'oxford_pets',
    'StanfordCars': 'stanford_cars',
    'SUN397': 'sun397',
    'UCF101': 'ucf101',
}

DS_TEMPLATE = {
    'Caltech101': 'a photo of a {}.',
    'DescribableTextures': 'a photo of a {} texture.',
    'EuroSAT': 'a centered satellite photo of a {}.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'Food101': 'a photo of a {}, a type of food.',
    'ImageNet': 'a photo of a {}.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'StanfordCars': 'a photo of a {}, a type of car.',
    'SUN397': 'a photo of a {}',
    'UCF101': 'a photo of a person doing {}.',
    "ImageNetV2": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'TransAgent',
                          "vision_depth": cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.TRANSAGENT.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.TRANSAGENT.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def normalize(images, dataset="ImageNet", type="mae/dino"):
    mean = torch.Tensor([0.485, 0.456, 0.406]) if dataset == "ImageNet" else torch.Tensor([0.5, 0.5, 0.5])
    std = torch.Tensor([0.229, 0.224, 0.225]) if dataset == "ImageNet" else torch.Tensor([0.5, 0.5, 0.5])
    mean = mean.to(images.device).view(1, 3, 1, 1).type_as(images)
    std = std.to(images.device).view(1, 3, 1, 1).type_as(images)
    return (images - mean) / std


def normalize_sam(images, type="sam"):
    mean = torch.Tensor([123.675, 116.28, 103.53]).to(images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.Tensor([58.395, 57.12, 57.375]).to(images.device).view(1, 3, 1, 1).type_as(images)
    return (images - mean) / std


def denormalize_clip(images, type="clip"):
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1).type_as(images)
    return images * std + mean


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, deep_prompts_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, deep_prompts_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_TEXT >= 1, "For TransAgent, Language prompt depth should be >=1"
        assert cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_VISION >= 1, "For TransAgent, Vision prompt depth should be >=1"
        self.prompt_depth_text = cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_TEXT
        self.prompt_depth_vision = cfg.TRAINER.TRANSAGENT.PROMPT_DEPTH_VISION
        n_ctx_text = cfg.TRAINER.TRANSAGENT.N_CTX_TEXT
        n_ctx_vision = cfg.TRAINER.TRANSAGENT.N_CTX_VISION
        ctx_init = cfg.TRAINER.TRANSAGENT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx_text) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx_text = n_ctx_text
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx_text, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx_text, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx_text)
        print('TransAgent design')
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx_text}")
        print(f"Number of context words (tokens) for Vision prompting: {n_ctx_vision}")
        self.ctx = nn.Parameter(ctx_vectors)

        # visual ctx random init
        vis_ctx_vectors = torch.empty(n_ctx_vision, 768, dtype=dtype)
        nn.init.normal_(vis_ctx_vectors, std=0.02)
        self.visual_ctx = nn.Parameter(vis_ctx_vectors)

        self.deep_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx_text, 512))
                                                   for _ in range(self.prompt_depth_text - 1)])  # except layer 0
        for single_prompt in self.deep_prompts_text:
            nn.init.normal_(single_prompt, std=0.02)
        self.deep_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx_vision, 768))
                                                     for _ in range(self.prompt_depth_vision - 1)])  # except layer 0
        for single_prompt in self.deep_prompts_vision:
            nn.init.normal_(single_prompt, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]

        if cfg.TRAINER.MODAL == "base2novel" or cfg.TRAINER.MODAL == "fewshot":
            prompt_template = "a photo of a {}"
        elif cfg.TRAINER.MODAL == "cross":
            prompt_template = DS_TEMPLATE[cfg.DATASET.NAME]
        else:
            raise ValueError(f"Not supported type: {cfg.TRAINER.MODAL}!")

        prompts = [prompt_template.format(name) for name in classnames]

        gpt_prompts_dict = {}
        vicuna_prompt_dict = {}
        excluded_list = ["ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"]
        if cfg.DATASET.NAME not in excluded_list:
            gpt_template_path = f"/home/ywguo/code/transagent/template_protext/{cfg.DATASET.NAME}_prompts.json"
            vicuna_template_path = f"/home/ywguo/code/transagent/template/{cfg.DATASET.NAME}_prompts.json"
            gpt_file = open(gpt_template_path, "r")
            vicuna_file = open(vicuna_template_path, "r")
            gpt_prompts_dict = json.load(gpt_file)
            gpt_prompts_dict = {k.lower().replace("_", " "): v for k, v in gpt_prompts_dict.items()}
            vicuna_prompt_dict = json.load(vicuna_file)
            vicuna_prompt_dict = {k.lower().replace("_", " "): v for k, v in vicuna_prompt_dict.items()}

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            llm_text_features = []
            gpt_text_features = []
            vicuna_text_features = []
            # zs_template_features = []
            if len(gpt_prompts_dict) > 0 and len(vicuna_prompt_dict) > 0:
                for single_key in classnames:
                    single_class_prompts_gpt = gpt_prompts_dict[single_key.lower().replace("_", " ")]
                    x_tokenized_gpt = torch.cat([clip.tokenize(p) for p in single_class_prompts_gpt])
                    text_features_gpt = clip_model_temp.encode_text(x_tokenized_gpt.cuda())
                    single_class_prompts_vicuna = vicuna_prompt_dict[single_key.lower().replace("_", " ")]
                    x_tokenized_vicuna = torch.cat([clip.tokenize(p) for p in single_class_prompts_vicuna])
                    text_features_vicuna = clip_model_temp.encode_text(x_tokenized_vicuna.cuda())
                    text_features = torch.cat([text_features_gpt, text_features_vicuna], dim=0)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    llm_text_features.append(text_features.mean(0).unsqueeze(0))
                    gpt_text_features.append(text_features_gpt.mean(0).unsqueeze(0))
                    vicuna_text_features.append(text_features_vicuna.mean(0).unsqueeze(0))
                llm_text_features = torch.cat(llm_text_features, dim=0)
                llm_text_features = llm_text_features / llm_text_features.norm(dim=-1, keepdim=True)
                gpt_text_features = torch.cat(gpt_text_features, dim=0)
                gpt_text_features = gpt_text_features / gpt_text_features.norm(dim=-1, keepdim=True)
                vicuna_text_features = torch.cat(vicuna_text_features, dim=0)
                vicuna_text_features = vicuna_text_features / vicuna_text_features.norm(dim=-1, keepdim=True)
            # for single_template in IMAGENET_TEMPLATES:
            #     x = [single_template.replace("{}", name) for name in classnames]
            #     x_tokenized = torch.cat([clip.tokenize(p) for p in x])
            #     text_features = clip_model_temp.encode_text(x_tokenized.cuda())
            #     zs_template_features.append(text_features.unsqueeze(1))
            zs_text_features = clip_model_temp.encode_text(tokenized_prompts.cuda())
            zs_text_features = zs_text_features / zs_text_features.norm(dim=-1, keepdim=True)

        self.llm_text_features = llm_text_features
        self.gpt_text_features = gpt_text_features
        self.vicuna_text_features = vicuna_text_features
        # self.fixed_embeddings = torch.cat(zs_template_features, dim=1).mean(dim=1)
        self.fixed_embeddings = zs_text_features
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx_text:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx_text
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        visual_ctx = self.visual_ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, visual_ctx, self.deep_prompts_text, self.deep_prompts_vision


def load_expert_knowledge(cfg, knowledge_path):
    suffix = 'train-' + 'shot_' + str(cfg.DATASET.NUM_SHOTS) + '-seed_' + str(cfg.SEED) + '.pkl'
    filepath = osp.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], knowledge_path, suffix)
    with open(filepath, "rb") as file:
        knowledge_dict = pickle.load(file)
    return knowledge_dict


# noisy top-k gating, modified from https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE.py
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, dtype=None):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts, dtype=dtype)
        self.noise_linear = nn.Linear(n_embed, num_experts, dtype=dtype)
        self.dtype = dtype

    def forward(self, input):
        logits = self.topkroute_linear(input)

        # Noise logits
        noise_logits = self.noise_linear(input)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class ConvRouter(nn.Module):

    def __init__(self, in_dim, out_dim, dtype):
        super(ConvRouter, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, dtype=dtype),
            nn.BatchNorm2d(out_dim, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1, dtype=dtype)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

        # regularization
        self.scl_text_weight = cfg.TRAINER.TRANSAGENT.TEXT_LOSS_WEIGHT
        self.scl_image_weight = cfg.TRAINER.TRANSAGENT.IMAGE_LOSS_WEIGHT
        self.scl_logits_weight = cfg.TRAINER.TRANSAGENT.LOGITS_LOSS_WEIGHT

        # multi-modal agent collaboration
        self.text_ln = self.text_encoder.ln_final
        self.text_proj = self.text_encoder.text_projection
        self.vision_ln = self.image_encoder.ln_post
        self.vision_proj = self.image_encoder.proj

        self.mm_temp = cfg.TRAINER.TRANSAGENT.MM_TEMP
        self.mm_loss_type = cfg.TRAINER.TRANSAGENT.MM_LOSS_TYPE
        self.mm_loss_weight = cfg.TRAINER.TRANSAGENT.MM_LOSS_WEIGHT
        if self.mm_loss_type == 'l1':
            self.mm_loss_func = nn.L1Loss()
        elif self.mm_loss_type == 'mse':
            self.mm_loss_func = nn.MSELoss()

        num_vl_experts = cfg.TRAINER.TRANSAGENT.NUM_VL_EXPERTS
        self.mm_gate = NoisyTopkRouter(self.n_cls * num_vl_experts, num_vl_experts, num_vl_experts, dtype=self.dtype)
        # self.mm_gate = ConvRouter(self.n_cls, num_vl_experts, dtype=self.dtype)

        # language agent collaboration
        num_llm = cfg.TRAINER.TRANSAGENT.NUM_LLM_EXPERTS
        self.llm_gate = NoisyTopkRouter(512 * num_llm, num_llm, num_llm, dtype=self.dtype)
        # self.llm_gate = ConvRouter(512, num_llm, dtype=self.dtype)
        self.llm_distill_weight = cfg.TRAINER.TRANSAGENT.LLM_LOSS_WEIGHT

        # vision agent collaboration
        vision_fusion_type = cfg.TRAINER.TRANSAGENT.VISION_FUSION_TYPE
        num_experts = cfg.TRAINER.TRANSAGENT.NUM_VISION_EXPERTS

        self.v_gate = None
        if vision_fusion_type == "gating":
            # self.v_gate = NoisyTopkRouter(768 * num_experts, num_experts, num_experts, dtype=self.dtype) # last-layer
            self.v_gate = nn.ModuleList([NoisyTopkRouter(768 * num_experts, num_experts, num_experts, dtype=self.dtype)
                                         for _ in range(cfg.TRAINER.TRANSAGENT.N_LAST_BLOCKS)])  # layer-wise
            # self.v_gate = nn.ModuleList([ConvRouter(768, num_experts, dtype=self.dtype)
            #                              for _ in range(cfg.TRAINER.TRANSAGENT.N_LAST_BLOCKS)])  # layer-wise

    def forward(self, image, label=None, batch_idx=0):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, visual_ctx, deep_prompts_text, deep_prompts_vision = self.prompt_learner()

        # compute the prompted features
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_prompts_text)
        image_features, feat_list = self.image_encoder(image.type(self.dtype), visual_ctx, deep_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            # prompted logits cross-entropy
            loss_ce = F.cross_entropy(logits, label)

            # multi-modal agent collaboration preprocess
            # process expert knowledge
            if self.cfg.TRAINER.MODAL == "base2novel":
                sd_knowledge = load_expert_knowledge(self.cfg, 'sd_knowledge')
                pixart_knowledge = load_expert_knowledge(self.cfg, "pixart_knowledge")
                blip2_knowledge = load_expert_knowledge(self.cfg, "blip2_knowledge")
                shikra_knowledge = load_expert_knowledge(self.cfg, "shikra_knowledge")
            elif self.cfg.TRAINER.MODAL == "cross":
                sd_knowledge = load_expert_knowledge(self.cfg, 'sd_knowledge_inall')
                pixart_knowledge = load_expert_knowledge(self.cfg, "pixart_knowledge_inall")
                blip2_knowledge = load_expert_knowledge(self.cfg, "blip2_knowledge_inall")
                shikra_knowledge = load_expert_knowledge(self.cfg, "shikra_knowledge_inall")
            elif self.cfg.TRAINER.MODAL == "fewshot":
                shots = self.cfg.DATASET.NUM_SHOTS
                seed = self.cfg.SEED
                sd_knowledge = load_expert_knowledge(self.cfg, f'sd_knowledge_{shots}shots_seed{seed}')
                pixart_knowledge = load_expert_knowledge(self.cfg, f"pixart_knowledge_{shots}shots_seed{seed}")
                blip2_knowledge = load_expert_knowledge(self.cfg, f"blip2_knowledge_{shots}shots_seed{seed}")
                shikra_knowledge = load_expert_knowledge(self.cfg, f"shikra_knowledge_{shots}shots_seed{seed}")
            else:
                raise ValueError(f"Not supported type: {self.cfg.TRAINER.MODAL}!")
            batch_size = image.shape[0]

            log_scores = sd_knowledge["log_scores"]
            log_scores_batch = []
            for scores in log_scores:
                scores = scores[batch_idx:batch_idx + batch_size].to(image.device)
                log_scores_batch.append(scores)
            log_scores = sum(log_scores_batch) / len(log_scores_batch)  # [bs, n_cls]

            log_scores2 = pixart_knowledge["log_scores"]
            log_scores_batch2 = []
            for scores in log_scores2:
                scores = scores[batch_idx:batch_idx + batch_size].to(image.device)
                log_scores_batch2.append(scores)
            log_scores2 = sum(log_scores_batch2) / len(log_scores_batch2)  # [bs, n_cls]

            blip2_logits = blip2_knowledge["logits_list"]
            blip2_logits_batch = blip2_logits[batch_idx:batch_idx + batch_size].to(image.device)  # [bs, n_cls]

            shikra_logits = shikra_knowledge["logits_list"]
            shikra_logits_batch = shikra_logits[batch_idx:batch_idx + batch_size].to(image.device)  # [bs, n_cls]

            # mm agent collaboration with prompt tokens similarity
            learned_scores = []
            learned_text_prompts = deep_prompts_text[-1]  # [n_ctx_text, dim]
            learned_text_prompts = learned_text_prompts.expand(self.n_cls, -1, -1).permute(1, 0, 2).half()
            learned_text_prompts = self.text_ln(learned_text_prompts).type(self.dtype)
            learned_text_prompts = learned_text_prompts @ self.text_proj

            learned_visual_prompts = deep_prompts_vision[-1]  # [n_ctx_vision, dim]
            learned_visual_prompts = learned_visual_prompts.expand(batch_size, -1, -1).permute(1, 0, 2).half()
            learned_visual_prompts = self.vision_ln(learned_visual_prompts)
            learned_visual_prompts = learned_visual_prompts @ self.vision_proj

            for tpt, vpt in zip(learned_text_prompts, learned_visual_prompts):
                tpt = tpt / tpt.norm(dim=-1, keepdim=True)
                vpt = vpt / vpt.norm(dim=-1, keepdim=True)
                learned_scores.append(logit_scale * vpt @ tpt.t())
            learned_scores = sum(learned_scores) / len(learned_scores)  # [bs, n_cls]

            mm_logits = torch.stack([log_scores, log_scores2, blip2_logits_batch, shikra_logits_batch],
                                    dim=1)  # [bs, num_vl_experts, n_cls]
            mm_logits_cat = torch.cat([log_scores, log_scores2, blip2_logits_batch, shikra_logits_batch],
                                      dim=1)  # [bs, num_vl_experts * n_cls]
            mm_weights, _ = self.mm_gate(mm_logits_cat)
            mm_weights = torch.softmax(mm_weights.unsqueeze(-1), dim=1)  # [bs, num_vl_experts, 1]
            mm_logits = torch.mean(mm_logits * mm_weights, dim=1)  # [bs, n_cls]

            temp = self.mm_temp
            if self.mm_loss_type == 'kl':
                loss_mm = F.kl_div(F.log_softmax(learned_scores / temp, dim=1),
                                   F.log_softmax(mm_logits / temp, dim=1),
                                   reduction='sum',
                                   log_target=True) * (temp * temp) / learned_scores.numel()

            else:
                loss_mm = self.mm_loss_func(input=learned_scores, target=mm_logits.half())

            loss_mm = self.mm_loss_weight * loss_mm

            # llm knowledge distillation
            # llm_text_features = self.prompt_learner.llm_text_features.type(self.dtype)

            # llm features gating fusion
            gpt_text_features = self.prompt_learner.gpt_text_features.type(self.dtype)
            vicuna_text_features = self.prompt_learner.vicuna_text_features.type(self.dtype)
            llm_text_features = torch.stack([gpt_text_features, vicuna_text_features], dim=1)  # [n_cls, num_llm, dim]
            llm_text_features_cat = torch.cat([gpt_text_features, vicuna_text_features],
                                              dim=-1)  # [n_cls, num_llm * dim]
            llm_weights, _ = self.llm_gate(llm_text_features_cat)
            llm_weights = torch.softmax(llm_weights.unsqueeze(-1), dim=1)  # [n_cls, num_llm, 1]
            llm_text_features = torch.mean(llm_text_features * llm_weights, dim=1)  # [n_cls, dim]

            loss_llm_dist = F.l1_loss(text_features, llm_text_features, reduction="mean")

            # zs features regularization
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))  # zs image features
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            loss_scl_text = F.l1_loss(text_features, fixed_embeddings.cuda())
            loss_scl_image = F.l1_loss(image_features, zero_shot_features.cuda())
            loss_scl_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()

            loss_scl = self.scl_logits_weight * loss_scl_logits + loss_scl_text * self.scl_text_weight + self.scl_image_weight * loss_scl_image

            total_loss = loss_ce + loss_mm + self.llm_distill_weight * loss_llm_dist + loss_scl

            return total_loss, feat_list

        return logits


@TRAINER_REGISTRY.register()
class TransAgent(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRANSAGENT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.TRANSAGENT.PREC == "fp32" or cfg.TRAINER.TRANSAGENT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.dtype = self.model.dtype

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name or "gate" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TRANSAGENT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.teacher_mae = vit_base(patch_size=16, num_classes=0)
        self.teacher_mae.to(self.device, self.dtype)
        self.teacher_mae.eval()
        load_experts(self.teacher_mae, "/home/ywguo/ckpt/MAE/mae_pretrain_vit_base.pth", model_name="mae")
        print("Model mae built.")

        self.teacher_dino = vit_base(patch_size=16, num_classes=0)
        self.teacher_dino.to(self.device, self.dtype)
        self.teacher_dino.eval()
        load_experts(self.teacher_dino, "/home/ywguo/ckpt/DINO/dino_vitbase16_pretrain.pth",
                     model_name="dino")
        print("Model dino built.")

        self.teacher_sam = sam_vit_base()
        self.teacher_sam.to(self.device, self.dtype)
        self.teacher_sam.eval()
        load_experts(self.teacher_sam, "/home/ywguo/ckpt/SAM/sam_vit_b.pth", model_name="sam")
        print("Model sam built.")

        self.teacher_vitdet = vitdet_base(patch_size=16, num_classes=0)
        self.teacher_vitdet.to(self.device, self.dtype)
        self.teacher_vitdet.eval()
        load_experts(self.teacher_vitdet, "/home/ywguo/ckpt/ViTDet/model_final_61ccd1.pkl",
                     model_name="vitdet")
        print("Model vitdet built.")

        self.vision_fusion_type = cfg.TRAINER.TRANSAGENT.VISION_FUSION_TYPE
        self.num_experts = cfg.TRAINER.TRANSAGENT.NUM_VISION_EXPERTS

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        v_gate = self.model.v_gate

        prec = self.cfg.TRAINER.TRANSAGENT.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss, feat_list = model(image, label, self.batch_idx)

            # vision agent collaboration
            image_sub = normalize(denormalize_clip(image), dataset=self.cfg.DATASET.NAME)
            image_sam = normalize_sam(denormalize_clip(image))
            n_last_blocks = self.cfg.TRAINER.TRANSAGENT.N_LAST_BLOCKS
            mae_feat = self.teacher_mae.get_intermediate_layers(image_sub.type(self.dtype), n_last_blocks)
            dino_feat = self.teacher_dino.get_intermediate_layers(image_sub.type(self.dtype), n_last_blocks)
            sam_feat = self.teacher_sam.get_intermediate_layers(image_sam.type(self.dtype), n_last_blocks)
            vitdet_feat = self.teacher_vitdet.get_intermediate_layers(image_sub.type(self.dtype), n_last_blocks)
            loss_dist = 0.
            vision_loss_weight = self.cfg.TRAINER.TRANSAGENT.VISION_LOSS_WEIGHT

            # layer-wise distillation
            for i in range(feat_list.shape[0]):
                clip_feat = feat_list[i]
                if self.vision_fusion_type == "average":
                    fused_feat = (mae_feat[i] + dino_feat[i] + sam_feat[i] + vitdet_feat[i]) / self.num_experts
                elif self.vision_fusion_type == "gating":
                    assert v_gate is not None
                    expert_feat = torch.stack([mae_feat[i], dino_feat[i], sam_feat[i], vitdet_feat[i]],
                                              dim=1)  # [bs, num_experts, seq_len, dim]
                    expert_feat_cat = torch.cat([
                        mae_feat[i].mean(1),
                        dino_feat[i].mean(1),
                        sam_feat[i].mean(1),
                        vitdet_feat[i].mean(1)],
                        dim=-1)  # [bs, num_experts * dim]
                    weights, _ = v_gate[i](expert_feat_cat)
                    weights = torch.softmax(weights.unsqueeze(-1).unsqueeze(-1), dim=1)  # [bs, num_experts, 1, 1]

                    fused_feat = torch.mean(expert_feat * weights, dim=1)  # [bs, seq_len, dim]

                else:
                    raise ValueError(f"Not supported fusion type: {self.vision_fusion_type}!")
                fused_feat = fused_feat / fused_feat.norm(dim=-1, keepdim=True)
                loss_dist += F.l1_loss(clip_feat, fused_feat)

            loss += vision_loss_weight * loss_dist
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            state_dict_keys = list(state_dict.keys())

            # remove gate in ckpt to prevent potential bugs in inference stage (due to imbalance split of base/novel classes)
            new_state_dict = OrderedDict()
            for key in state_dict_keys:
                if "gate" not in key:
                    new_state_dict[key] = state_dict[key]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            # self._models[name].load_state_dict(state_dict, strict=False)
            self._models[name].load_state_dict(new_state_dict, strict=False)
