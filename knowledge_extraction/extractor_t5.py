import os
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import time
import datetime
import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse

from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL

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

from chat_templates import PROMPT


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


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

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

    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=f'{args.pretrained_models_dir}')
    t5_save_dir = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], cfg.OUTPUT_DIR)
    os.makedirs(t5_save_dir, exist_ok=True)

    dataset = eval(cfg.DATASET.NAME)(cfg)
    classnames = dataset.classnames
    classnames = [name.replace("_", " ") for name in classnames]
    print("number of classes: ", dataset.num_classes)
    template = DS_TEMPLATE[cfg.DATASET.NAME]
    prompts = [template.format(name) for name in classnames]

    with torch.no_grad():

        save_path = os.path.join(t5_save_dir, f"{cfg.DATASET.NAME}_t5f_seed{cfg.SEED}.npz")
        try:
            caption_emb, emb_mask = t5.get_text_embeddings(prompts)
            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
            print(f"Feature extraction finished for {cfg.DATASET.NAME}!")
        except Exception as e:
            print(e)


if __name__ == '__main__':
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
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--pretrained_models_dir', default='', type=str)
    args = parser.parse_args()
    main(args)
