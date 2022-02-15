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
from model.dcgan import Generator as GeneratorDCGAN, Discriminator as DiscriminatorDCGAN
from model.began import BEGAN_Decoder
from functools import reduce

def dict_to_str(d, exclude=None):
    s = []
    for k, v in d.items():
        if exclude is not None and k in exclude:
            continue
        s.append(f"{k}={v}")
    return ".".join(s)


def str_to_dict(s):
    blocks = s.split('.')
    clean_blocks = []
    for b in blocks:
        if '=' in b:
            clean_blocks.append(b)
        else:
            clean_blocks[-1] += ('.' + b)

    d = {}
    for b in clean_blocks:
        k, v = b.split('=')
        d[k] = v

    return d


def _gen_img(img):
    plt.figure(figsize=(16, 9))
    plt.imshow(img)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


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


def parse_images_folder(p):
    p = Path(p)
    _, _, split, image_name, img_size = p.parts
    return split, image_name, img_size


# Use get_results_folder for all models, use dummy n_cuts if necessary
def get_results_folder(image_name, model, cuts, dataset, forward_model,
                       recovery_params, base_dir):
    return (Path(base_dir) / 'results' / model / dataset / image_name / str(forward_model) / recovery_params / f'cuts={cuts}' )


def forward_model_from_str(s):
    lst = s.split('.', 1)
    if len(lst) > 1:
        name = lst[0]
        params = str_to_dict(lst[1])
        return name, params
    else:
        name = lst[0]
        return name, {}

    
## New code
def _rename_state_dict(name_mapping, state_dict):
    # rename layers
    for key in list(state_dict.keys()):
        layer = key.split('.')[0]
        rest = key.split('.')[1]
        new_key = name_mapping[layer] + '.' + rest
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
      
    return state_dict


def _load_state_dcgan(state_dict, model, name_mapping, skip_bn1: bool = False):
  for i in range(4):
    if i == 0 and skip_bn1:
      continue
    del state_dict[f'bn{i+1}.num_batches_tracked']

  new_state_dict = _rename_state_dict(name_mapping, state_dict)
  model.load_state_dict(state_dict)


def load_pretrained_dcgan_gen(state_dict):
  params = state_dict['params']
  gen = GeneratorDCGAN(params['nz'], params['ngf'], params['nc'])
  old_to_new = {
      'tconv1': 'main.0.0',
      'bn1': 'main.0.1',
      'tconv2': 'main.1.0',
      'bn2': 'main.1.1',
      'tconv3': 'main.2.0',
      'bn3': 'main.2.1',
      'tconv4': 'main.3.0',
      'bn4': 'main.3.1',
      'tconv5': 'main.4.0'
  }
  _load_state_dcgan(state_dict['generator'], gen, old_to_new)
  return gen


def load_pretrained_dcgan_disc(state_dict):
  params = state_dict['params']
  disc = DiscriminatorDCGAN(params['nc'], params['ndf'])
  old_to_new = {
      'conv1': 'main.0',
      'conv2': 'main.2',
      'bn2': 'main.3',
      'conv3': 'main.5',
      'bn3': 'main.6',
      'conv4': 'main.8',
      'bn4': 'main.9',
      'conv5': 'main.11',
  }
  _load_state_dcgan(state_dict['discriminator'], disc, old_to_new, True)
  return disc

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

def load_pretrained_began_disc(state_dict):
    disc = BEGAN_Discriminator()
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
    disc.load_state_dict(new_state_dict)
    return disc


class ImgDataset(Dataset):
    def __init__(self, img_dir, img_list, img_size):
        self.img_dir = img_dir
        self.img_size = img_size
        
        with open(img_list) as file:
            self.image_names = [line.rstrip() for line in file]
                               
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_tensor = load_target_image(os.path.join(self.img_dir, img_name), self.img_size)
        return img_tensor, img_name


def setup_logger(name, log_dest):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_dest)
    fh.setFormatter(fmt)
    # only write to log file if we have a warning
    fh.setLevel(logging.WARNING)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_logs_folder(base_dir, project_name, forward_model, metadata_str):
    dir_path = Path(base_dir) / project_name / str(forward_model)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path / f'{metadata_str}_logs.txt'


ROOT_LOGGER_NAME = 'restore_logger'


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


##


if __name__ == '__main__':
    parse_results_folder('./runs/results')
    # parse_baseline_results_folder('./final_runs/baseline_results')
