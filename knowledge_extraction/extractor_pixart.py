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
import numpy as np

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

from diffusers.models import AutoencoderKL
from diffusion import build_model, load_checkpoint
from diffusion.utils.misc import read_config
from imagenet_templates import IMAGENET_TEMPLATES



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


def exists(val):
    return val is not None


def register_attention_control(model, controller):
    def ca_forward(self):
        def forward(x, cond, mask=None):
            h = self.num_heads
            B, N, C = x.shape

            q = self.q_linear(x).reshape(B, N, h, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv_linear(cond).reshape(B, cond.shape[1], 2, h, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k)

            # if exists(mask):
            #     mask = rearrange(mask, 'b ... -> b (...)')
            #     max_neg_value = -torch.finfo(sim.dtype).max
            #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
            #     sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2)  # we attain cross attention here

            x = einsum('b i j, b j d -> b i d', attn, v)
            x = rearrange(x, '(b h) n d -> b n (h d)', h=h)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count):
        if net_.__class__.__name__ == 'MultiHeadCrossAttention':
            net_.forward = ca_forward(net_)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()

    for net in sub_nets:
        cross_att_count += register_recr(net[1], 0)

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
    def forward(self, attn):
        raise NotImplementedError

    def __call__(self, attn):
        attn = self.forward(attn)
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
        return {"attn_all": []}

    def forward(self, attn):
        key = "attn_all"
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

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_forward_control(model):
    self = model

    def forward(x, t, y, mask=None):

        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (n_cls, 120, C) tensor of class labels
        """
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1)
            y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[1]] * y.shape[0]
            # y = y.view(1, -1, x.shape[-1])
        for block in self.blocks:
            # x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
            x = block(x, y, t0)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    self.forward = forward


class PixArtWrapper(nn.Module):
    def __init__(self, model, use_attn=True) -> None:
        super().__init__()
        self.model = model
        self.attention_store = AttentionStore()
        self.size32 = 32
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(model, self.attention_store)
        register_forward_control(model)

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        _ = self.model(*args, **kwargs)  # we do not need deep features
        cross_attn_out_list = []
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn32 = self.process_attn(avg_attn)
            cross_attn_out_list.append(attn32)
        return cross_attn_out_list

    def process_attn(self, avg_attn):
        attns = {self.size32: []}
        for attn in avg_attn['attn_all']:
            size = int(math.sqrt(attn.shape[1]))
            # attns[size].append(rearrange(attn, 'b (h w) c -> b c h w', h=size))
            attns[size].append(attn)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        return attn32


def vqvae_denormalize(images, type='vqvae'):
    images = 2 * images - 1
    return images


def get_sd_latent(image, encoder_vq, scale=0.18215):
    image = vqvae_denormalize(image)
    with torch.no_grad():
        latent = encoder_vq.encode(image).latent_dist
    latent = scale * latent.mode().detach()
    return latent.type(image.dtype)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.config:
        cfg.MODEL.CONFIG = args.config

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
    cfg.MODEL.NAME = "pixart"
    cfg.MODEL.CONFIG = "/path/to/transagent/configs/pixart/PixArt_xl2_img512.py"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.MODEL.MODAL = "base2novel"


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
    pixart_config = read_config(cfg.MODEL.CONFIG)
    image_size = pixart_config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(pixart_config, 'pred_sigma', True)
    learn_sigma = getattr(pixart_config, 'learn_sigma', True) and pred_sigma
    model_kwargs = {"window_block_indexes": pixart_config.window_block_indexes, "window_size": pixart_config.window_size,
                    "use_rel_pos": pixart_config.use_rel_pos, "lewei_scale": pixart_config.lewei_scale}
    pixart_model = build_model(pixart_config.model,
                               pixart_config.grad_checkpointing,
                               pixart_config.get('fp32_attention', False),
                               input_size=latent_size,
                               learn_sigma=learn_sigma,
                               pred_sigma=pred_sigma,
                               **model_kwargs)
    msg = load_checkpoint(pixart_config.load_from, pixart_model, load_ema=pixart_config.get('load_ema', False))
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pixart_config.load_from, msg))

    encoder_vq = AutoencoderKL.from_pretrained(pixart_config.vae_pretrained).cuda()
    model = PixArtWrapper(pixart_model)
    model.cuda()
    model.eval()

    ###################################################################################################################
    # Start Feature Extractor
    if cfg.MODEL.MODAL == "base2novel":
        t5_feat_path = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], "t5_text_feat_wmask_base",
                                    f"{cfg.DATASET.NAME}_t5f_seed{cfg.SEED}.npz")
    elif cfg.MODEL.MODAL == "cross" or cfg.MODEL.MODAL == "fewshot":
        t5_feat_path = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], "t5_text_feat_wmask", f"{cfg.DATASET.NAME}_t5f.npz")
    else:
        raise ValueError(f"Unsupported modal: {cfg.MODEL.MODAL}")
    t5_feat = np.load(t5_feat_path)

    text_feat = torch.from_numpy(t5_feat['caption_feature']).cuda()  # [n_cls, n_ctx. dim]

    log_scores = []
    dataiter = iter(data_loader)
    for step in range(1, len(dataiter) + 1):
        batch = next(dataiter)
        image = batch["img"].cuda()

        with torch.no_grad():
            latent = get_sd_latent(image, encoder_vq)
            y = text_feat.mean(1)  # [n_cls, dim]
            y = y / y.norm(dim=-1, keepdim=True)
            y = y.expand(latent.shape[0], -1, -1)

            t = torch.ones((latent.shape[0],), device=latent.device).long()
            cross_attn_batch = model(latent, t, y)  # [layer_size, batch_size, *]
            if step == 1:
                for item in cross_attn_batch:
                    log_score = torch.logsumexp(item, -2)  # along the tokens axis
                    log_scores.append(log_score.detach().cpu().half())
            else:
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
