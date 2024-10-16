import os
import torch
from collections import OrderedDict

if __name__ == "__main__":
    path = "/path/to/sam_vit_b_01ec64.pth"
    state_dict = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "image_encoder." in k:
            new_state_dict[k[14:]] = v
    torch.save(new_state_dict, "/path/to/checkpoint/dir/sam_vit_b.pth")

