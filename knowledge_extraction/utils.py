import os
import torch
import math
import pickle
from collections import OrderedDict

from ldm.util import instantiate_from_config


def load_pretrained_weights(model, pretrained_weights, checkpoint_key="teacher", model_name="dino"):
    if os.path.isfile(pretrained_weights):
        if model_name == "vitdet":
            state_dict = OrderedDict()
            with open(pretrained_weights, "rb") as file:
                state_dict_ckpt = pickle.load(file)
            state_dict_ckpt = state_dict_ckpt["model"]
            for k, v in state_dict_ckpt.items():
                if "backbone.net." in k and "rel" not in k:
                    k = k.replace("backbone.net.", "")
                    state_dict[k] = torch.tensor(v)
        else:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
        if model_name == "dino":
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        elif model_name == "mae":
            state_dict = state_dict['model']
        elif model_name == "sam":
            interpolate_pos_embed(model, state_dict)

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        raise ValueError("No pretrained weights under folder.")


def load_sd_from_config(config, ckpt, verbose=False):
    print(f"Loading stable diffusion from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def interpolate_pos_embed(model, checkpoint_model):
    for k in checkpoint_model.keys():
        if 'pos_embed' in k:
            pos_embed_checkpoint = checkpoint_model[k]

            # height (== width) for the checkpoint position embedding
            orig_size = pos_embed_checkpoint.shape[1]
            # height (== width) for the new position embedding
            new_size = model.pos_embed.shape[1]

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))

                pos_tokens = pos_embed_checkpoint.permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1)
                checkpoint_model[k] = pos_tokens
        if "rel_pos" in k:
            rel_pos_embed_checkpoint = checkpoint_model[k]
            orig_size = rel_pos_embed_checkpoint.shape[0]
            new_size = 2 * 14 - 1

            if orig_size != new_size:
                print("Relative position interpolate from %d to %d" % (orig_size, new_size))
                rel_pos_resized = rel_pos_embed_checkpoint.reshape(1, orig_size, -1).permute(0, 2, 1)
                rel_pos_resized = torch.nn.functional.interpolate(
                    rel_pos_resized, size=new_size, mode='linear'
                )
                rel_pos_resized = rel_pos_resized.reshape(-1, new_size).permute(1, 0)
                checkpoint_model[k] = rel_pos_resized
