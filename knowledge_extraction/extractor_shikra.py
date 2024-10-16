from omegaconf import OmegaConf
import os
import argparse
import pickle
import json

import torch
from torch import nn
from torchvision import transforms

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

from mmengine.config import Config
import transformers
from models.shikra import ShikraLlamaForCausalLM

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
    cfg.MODEL.NAME = "shikra"
    cfg.MODEL.CONFIG = "/path/to/transagent/configs/shikra/shikra.py"
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
    shikra_config = Config.fromfile(cfg.MODEL.CONFIG)
    model_args = shikra_config.model_args
    shikra_model = ShikraLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16
    )
    shikra_model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model_vision_dict = shikra_model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )

    dtype = torch.float16
    device = "cuda"

    shikra_model.model.vision_tower[0].to(dtype=dtype, device=device)
    vision_config = model_vision_dict['vision_config']
    shikra_model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    shikra_model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                             tokenizer=tokenizer,
                                             device=device
                                             )
    print(f"Model {cfg.MODEL.NAME} built.")

    shikra_model.cuda()
    shikra_model.eval()

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

    logits_list = []
    dataiter = iter(data_loader)
    for step in range(1, len(dataiter) + 1):
        batch = next(dataiter)
        image = batch["img"].cuda()
        image = image.to(device, torch.float16)

        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            vision_tower = shikra_model.model.vision_tower[0]
            mm_projector = shikra_model.model.mm_projector

            image_forward_outs = vision_tower(image, output_hidden_states=True)
            select_hidden_state_layer = getattr(shikra_config, "mm_vision_select_layer", -1)
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:]
            image_features = mm_projector(image_features)

            outputs = shikra_model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            pooled_img_feat = image_features.mean(dim=1)  # [bs, dim]
            pooled_img_feat = pooled_img_feat / pooled_img_feat.norm(dim=1, keepdim=True)
            text_feat = outputs["hidden_states"][-1].mean(dim=1)  # [n_cls, dim]
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            logits = pooled_img_feat @ text_feat.t()

        logits_list.append(logits)
        if step % 20 == 0:
            print(f"[{step}/{len(dataiter)}]")
    logits_list = torch.cat(logits_list, dim=0)
    save_dir = os.path.join(cfg.DATASET.ROOT, DS_PATH[cfg.DATASET.NAME], cfg.OUTPUT_DIR)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = os.path.join(save_dir, f"{args.split}-shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}.pkl")
    data = {"logits_list": logits_list}
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
