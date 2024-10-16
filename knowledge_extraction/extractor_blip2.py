from omegaconf import OmegaConf
import os
import argparse
import pickle
import json

import torch
from torch import nn
from torchvision import transforms

from transformers import AutoTokenizer, Blip2Model

import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

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
    cfg.MODEL.NAME = "blip2"
    cfg.MODEL.CKPT = "/path/to/blip2-opt-2.7b"
    cfg.MODEL.SIMPLE_TEMPLATE = False
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

    device = "cpu"
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
        device = "cuda"
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.CKPT)

    model = Blip2Model.from_pretrained(
        cfg.MODEL.CKPT, torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    print(f"Model {cfg.MODEL.NAME} built.")

    ###################################################################################################################
    # Start Knowledge Extractor
    classnames = dataset.classnames
    classnames = [name.replace("_", " ") for name in classnames]

    # hand-crafted templates
    if cfg.MODEL.SIMPLE_TEMPLATE:
        template = "a photo of a {}."
    else:
        template = DS_TEMPLATE[cfg.DATASET.NAME]

    prompts = [template.format(name) for name in classnames]

    mm_feat_list = []
    logits_list = []
    dataiter = iter(data_loader)
    for step in range(1, len(dataiter) + 1):
        batch = next(dataiter)
        image = batch["img"]
        image = image.to(device, torch.float16)

        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            mm_feat = model.get_qformer_features(pixel_values=image, return_dict=True)
            pooled_feat = mm_feat["pooler_output"]  # [bs, dim]
            pooled_feat_proj = model.language_projection(pooled_feat)
            pooled_feat_proj = pooled_feat_proj / pooled_feat_proj.norm(dim=1, keepdim=True)
            text_feat = model.get_text_features(input_ids, output_hidden_states=True, return_dict=True)
            last_text_feat = text_feat["hidden_states"][-1].mean(dim=1)  # [n_cls, dim]
            last_text_feat = last_text_feat / last_text_feat.norm(dim=1, keepdim=True)
            logits = pooled_feat_proj @ last_text_feat.t()

        mm_feat_list.append(pooled_feat)
        logits_list.append(logits)
        if step % 20 == 0:
            print(f"[{step}/{len(dataiter)}]")
    mm_feat_list = torch.cat(mm_feat_list, dim=0)
    logits_list = torch.cat(logits_list, dim=0)
    save_dir = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], cfg.OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)

    save_filename = os.path.join(save_dir, f"{args.split}-shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pkl")
    data = {"mm_feat_list": mm_feat_list, "logits_list": logits_list}
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
