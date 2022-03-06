import io
import os
import pickle
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from scipy.stats import truncnorm
from torchvision import transforms
from model.began import BEGAN_Decoder
from functools import reduce

## From this point onwards to the next line comment are functions from the original GS repo
## https://github.com/nik-sm/generator-surgery/blob/master/utils.py
def dict_to_str(d, exclude=None):
    s = []
    for k, v in d.items():
        if exclude is not None and k in exclude:
            continue
        s.append(f"{k}={v}")
    return ".".join(s)


def load_target_image(img, target_size):
    if img.endswith('.pt'):
        x = torch.load(img)
    else:
        image = Image.open(img)
        height, width = image.size

        if height > width:
            crop = transforms.CenterCrop((width, width))
        else:
            crop = transforms.CenterCrop((height, height))

        t = transforms.Compose([
            crop,
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])
        x = t(image)
    return x


def psnr(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)

    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    mse = F.mse_loss(img1, img2)
    return psnr_from_mse(mse)


def psnr_from_mse(mse):
    if mse == 0:
        return -1
    pixel_max = torch.tensor(1.0)
    p = 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)
    if isinstance(p, torch.Tensor):
        p = p.item()
    return p


def get_z_vector(shape, mode, limit=1, **kwargs):
    if mode == 'normal':
        z = torch.randn(*shape, **kwargs) * limit
    elif mode == 'clamped_normal':
        # Clamp between -truncation, truncation
        z = torch.clamp(torch.randn(*shape, **kwargs), -limit, limit)
    elif mode == 'truncated_normal':
        # Resample if any point lands outside -limit, limit
        values = truncnorm.rvs(-2, 2, size=shape).astype(np.float32)
        z = limit * torch.from_numpy(values).to(kwargs['device'])
        # raise NotImplementedError()
    elif mode == 'rectified_normal':
        # Max(N(0,1), 0)
        raise NotImplementedError()
    elif mode == 'uniform':
        z = 2 * limit * torch.rand(shape, **kwargs) - limit
    elif mode == 'zero':
        z = torch.zeros(*shape, **kwargs)
    else:
        raise NotImplementedError()
    return z


def get_images_folder(dataset, image_name, img_size, base_dir):
    return Path(base_dir) / 'images' / dataset / image_name / str(img_size)

def get_results_folder(image_name, model, cuts, dataset, forward_model,
                       recovery_params, base_dir):
    return (Path(base_dir) / 'results' / model / dataset / image_name / str(forward_model) / recovery_params / f'cuts={cuts}' )



## New functions

def _rename_state_dict(name_mapping, state_dict):
    for k in list(state_dict.keys()):
        ksplits = k.split('.')
        nk = name_mapping[ksplits[0]] + '.' + ksplits[1]
        state_dict[nk] = state_dict[k]
        del state_dict[k]
    return state_dict

def load_pretrained_began_gen(state_dict):
    gen = BEGAN_Decoder()
    old_to_new = {
        'l0': 'layers.0.0', # linear
        'l1': 'layers.1.net.0',
        'l2': 'layers.2.net.0',
        'l3': 'layers.4.net.0',
        'l4': 'layers.5.net.0',
        'l5': 'layers.7.net.0',
        'l6': 'layers.8.net.0',
        'l7': 'layers.10.net.0',
        'l8': 'layers.11.net.0',
        'l9': 'layers.15.0',
        'l10': 'layers.13.net.0',
        'l11': 'layers.14.net.0',
    }
    new_state_dict = _rename_state_dict(old_to_new, state_dict)
    gen.load_state_dict(new_state_dict)
    return gen

class ImgListDs(Dataset):
    def __init__(self, img_dir, img_list, img_size):
        self.img_dir = img_dir
        self.img_size = img_size
        file = open(img_list)
        self.image_names = [line.rstrip() for line in file]
        file.close()
                               
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_tensor = load_target_image(os.path.join(self.img_dir, img_name), self.img_size)
        return img_tensor, img_name 


def calc_num_params(first_cut:int, second_cut:int, z_number:int, model='began'):
    if model == 'began':
        gen = BEGAN_Decoder()
    else:
        assert False, 'Not implemented'
    
    num_params = 0
    
    z1_dim, z1_dim2 = gen.input_shapes[first_cut]
    
    num_params += reduce(lambda x,y: x*y, z1_dim) * z_number
    
    if len(z1_dim2) > 0:
        num_params += reduce(lambda x,y: x*y, z1_dim2) * z_number
    
    if z_number > 1:
        assert second_cut != -1
        # uses multiple code
        intermed_dim, z2_dim = gen.input_shapes[second_cut]
        
        # add the weights
        num_params += z_number * intermed_dim[0] # num channels
        
        if len(z2_dim) > 0:
            num_params += (z2_dim[0] * z2_dim[1] * z2_dim[2])
    
    return num_params


