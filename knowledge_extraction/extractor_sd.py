from omegaconf import OmegaConf
import os
import argparse
import pickle
import json

import torch as th
import torch
import math
import abc
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from inspect import isfunction

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet

from dassl.utils import set_random_seed
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper

from utils import load_sd_from_config
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip import clip
from imagenet_templates import IMAGENET_TEMPLATES

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
    'Caltech101': "a photo of a {}.",
    'DescribableTextures': "a photo of a {} texture.",
    'EuroSAT': "a centered satellite photo of a {}.",
    'FGVCAircraft': "a photo of a {}, a type of aircraft.",
    'Food101': "a photo of a {}, a type of food.",
    'ImageNet': "a photo of a {}.",
    'OxfordFlowers': "a photo of a {}, a type of flower.",
    'OxfordPets': "a photo of a {}, a type of pet.",
    'StanfordCars': "a photo of a {}, a type of car.",
    'SUN397': "a photo of a {}.",
    'UCF101': "a photo of a person doing {}.",
}


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)  # we attain cross attention here

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.diffusion_model.named_children()

    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        if max_size is None:
            self.max_size = self.base_size // 2
        else:
            self.max_size = max_size


def register_hier_output(model):
    self = model.diffusion_model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding

    def forward(x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # import pdb; pdb.set_trace()
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        out_list = []

        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)

        out_list.append(h)
        return out_list

    self.forward = forward


class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None,
                 attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        register_hier_output(unet)
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        feat_out_list = self.unet(*args, **kwargs)  # we do not need deep features
        cross_attn_out_list = []
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn)
            # out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            # out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            cross_attn_out_list.append(attn16)
            cross_attn_out_list.append(attn32)
            if attn64 is not None:
                # out_list[3] = torch.cat([out_list[3], attn64], dim=1)
                cross_attn_out_list.append(attn64)
        return feat_out_list, cross_attn_out_list

    def process_attn(self, avg_attn):
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                # attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
                attns[size].append(up_attn)
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64


def vqvae_denormalize(images, type='vqvae'):
    images = 2 * images - 1
    return images


def get_sd_latent(image, encoder_vq, scale=0.18215):
    image = vqvae_denormalize(image)
    with torch.no_grad():
        latent = encoder_vq.encode(image)
    latent = scale * latent.mode().detach()
    return latent.type(image.dtype)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.config:
        cfg.MODEL.CONFIG = args.config

    if args.ckpt:
        cfg.MODEL.CKPT = args.ckpt

    if args.seed:
        cfg.SEED = args.seed


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.MODEL = CN()
    cfg.MODEL.NAME = "sd"
    cfg.MODEL.CONFIG = "/path/to/transagent/configs/stable-diffusion/v1-inference.yaml"
    cfg.MODEL.CKPT = "/path/to/sd-v1-4.ckpt"
    cfg.MODEL.SIMPLE_TEMPLATE = False
    cfg.MODEL.LLM_TEMPLATE = False
    cfg.MODEL.ENSEMBLE_TEMPLATE = False
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    ######################################
    #   Setup DataLoader
    ######################################
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, dataset_input, transform=tfm_train, is_train=False),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    ########################################
    #   Setup Network
    ########################################
    sd_config = OmegaConf.load(cfg.MODEL.CONFIG)
    sd_model = load_sd_from_config(sd_config, cfg.MODEL.CKPT)

    encoder_vq = sd_model.first_stage_model
    unet = UNetWrapper(sd_model.model, max_attn_size=64)
    sd_model.model = None
    sd_model.first_stage_model = None

    ###################################################################################################################
    # Start Feature Extractor
    classnames = dataset.classnames
    classnames = [name.replace("_", " ") for name in classnames]

    # hand-crafted templates
    if cfg.MODEL.SIMPLE_TEMPLATE:
        template = "a photo of a {}."
    else:
        template = DS_TEMPLATE[cfg.DATASET.NAME]

    prompts = [template.format(name) for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

    c_crossattn = []
    assert not (cfg.MODEL.LLM_TEMPLATE and cfg.MODEL.ENSEMBLE_TEMPLATE)
    # LLM-generated templates
    if cfg.MODEL.LLM_TEMPLATE:
        print("Using LLM generated templates!")
        file = open(f"../template/{cfg.DATASET.NAME}_prompts.json", "r")
        gpt_prompts_dict = json.load(file)
        gpt_prompts_dict = {k.lower().replace("_", " "): v for k, v in gpt_prompts_dict.items()}
        for single_key in classnames:
            single_class_prompts = gpt_prompts_dict[single_key.lower().replace("_", " ")]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            class_features = sd_model.get_learned_conditioning(single_class_prompts)
            class_features = class_features[torch.arange(class_features.shape[0]), tokenized_prompts.argmax(dim=-1)]
            class_feature = torch.mean(class_features, dim=0)  # inner-class mean
            c_crossattn.append(class_feature)
        c_crossattn = torch.stack(c_crossattn, dim=0)
    # ensemble templates
    elif cfg.MODEL.ENSEMBLE_TEMPLATE:
        print("Using ensemble of hand-crafted templates!")
        total_templates = IMAGENET_TEMPLATES
        for i, temp in enumerate(total_templates):
            temp_prompts = [temp.format(c) for c in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in temp_prompts])
            temp_features = sd_model.get_learned_conditioning(temp_prompts)
            temp_features = temp_features[torch.arange(temp_features.shape[0]), tokenized_prompts.argmax(dim=-1)]
            c_crossattn.append(temp_features)
        c_crossattn = torch.stack(c_crossattn, dim=0).mean(dim=0)  # inter-template mean

    if not torch.is_tensor(c_crossattn):
        print("Using hand-crafted templates!")
        c_crossattn = sd_model.get_learned_conditioning(prompts)  # [n_cls, n_ctx, dim]
        c_crossattn = c_crossattn[torch.arange(c_crossattn.shape[0]), tokenized_prompts.argmax(dim=-1)]

    c_crossattn = c_crossattn / c_crossattn.norm(dim=-1, keepdim=True)

    # feat_list = []
    log_scores = []
    dataiter = iter(data_loader)
    for step in range(1, len(dataiter) + 1):
        batch = next(dataiter)
        image = batch["img"].cuda()

        with torch.no_grad():
            latent = get_sd_latent(image, encoder_vq)
            text_feat = c_crossattn.expand(latent.shape[0], -1, -1)

            t = torch.ones((latent.shape[0],), device=latent.device).long()
            feat_batch, cross_attn_batch = unet(latent, t, c_crossattn=[text_feat])  # [layer_size, batch_size, *]
            if step == 1:
                # for item in feat_batch:
                #     feat = F.adaptive_max_pool2d(item, (1, 1))
                #     feat = feat.view(feat.shape[0], -1)
                #     feat_list.append(feat.detach().cpu().half())
                for item in cross_attn_batch:
                    log_score = torch.logsumexp(item, -2)  # along the tokens axis
                    log_scores.append(log_score.detach().cpu().half())
            else:
                # for i in range(len(feat_list)):
                #     temp_feat = feat_list[i]  # [bs, c]
                #     feat = feat_batch[i]
                #     feat = F.adaptive_max_pool2d(feat, (1, 1))
                #     feat = feat.view(feat.shape[0], -1)
                #     new_feat = torch.cat([temp_feat, feat.detach().cpu().half()], dim=0)
                #     feat_list[i] = new_feat
                for i in range(len(log_scores)):
                    temp_score = log_scores[i]  # [bs, n_cls]
                    attn = cross_attn_batch[i]
                    score = torch.logsumexp(attn, -2)  # along the tokens axis
                    new_score = torch.cat([temp_score, score.detach().cpu().half()], dim=0)
                    log_scores[i] = new_score
        if step % 20 == 0:
            print(f"[{step}/{len(dataiter)}]")
    save_dir = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], cfg.OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = os.path.join(save_dir, f"{args.split}-shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pkl")
    # data = {"feat_list": feat_list, "log_scores": log_scores}
    data = {"log_scores": log_scores}
    with open(save_filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Feature extraction finished for {cfg.DATASET.NAME}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--config", type=str, default="", help="path to config file")
    parser.add_argument("--ckpt", type=str, default="", help="path to checkpoint file")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
