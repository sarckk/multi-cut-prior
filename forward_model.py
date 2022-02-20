"""
For each image recovery task, we define an explicit forward model.

NOTE - for most forward models, we treat each number in the image
as a "pixel", so it has H x W x Ch total pixels.
"""
import math
import sys
from abc import ABC, abstractmethod
from typing import Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import load_target_image

DEFAULT_DEVICE = 'cuda:0'


class ForwardModel(ABC):
    viewable = False

    @abstractmethod
    def __call__(self, img):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NoOp(ForwardModel):
    viewable = True

    def __init__(self, **fm_kwargs):
        pass

    def __call__(self, img):
        return img

    def __str__(self):
        return 'NoOp'


def get_random_mask(img_shape: Tuple[int, int, int], fraction_kept: float = None, n_kept: int = None, device=None):
    """
    For image of shape CHW, returns random boolean
    mask of the same shape.
    """
    if n_kept is None and fraction_kept is None:
        raise ValueError()

    n_pixels = np.prod(img_shape)
    if fraction_kept:
        n_kept = int(fraction_kept * n_pixels)

    mask = torch.zeros(img_shape)
    if device:
        mask = mask.to(device)

    random_coords = torch.randperm(int(n_pixels))
    for i in range(n_kept):
        random_coord = np.unravel_index(random_coords[i], img_shape)
        mask[random_coord] = 1
    return mask


# from IAGAN repo:
def rand_mask(size, thresh):
    half_size = np.floor(size / 2).astype('int32')
    idxX = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size)) * thresh), half_size)
    idxY = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size)) * thresh), half_size)
    mask_t = torch.zeros(size, size)
    mask_t[idxY, idxX] = 1
    # Duplicate
    dupIdx = [i for i in range(half_size - 1, -1, -1)]
    mask_t[:half_size, half_size:] = mask_t[:half_size, dupIdx]  # flip x
    mask_t[half_size:, :half_size] = mask_t[dupIdx, :half_size]  # flip y
    x, y = np.meshgrid(dupIdx, dupIdx)
    mask_t[half_size:, half_size:] = mask_t[y, x]  # flip x and y
    mask = np.array(mask_t)

    ratio = np.sum(mask == 1) / mask.size
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).unsqueeze(4)
    mask_t = torch.cat((mask_t, mask_t), 4)
    return mask_t, ratio


class InpaintingScatter(ForwardModel):
    """
    Mask random pixels
    """
    viewable = True
    inverse = False

    def __init__(self, img_shape, fraction_kept, device=DEFAULT_DEVICE):
        """
        img_shape - 3 x H x W
        fraction_kept - number in [0, 1], what portion of pixels to retain
        """
        assert fraction_kept <= 1 and fraction_kept >= 0
        self.fraction_kept = fraction_kept
        self.A = get_random_mask(img_shape, fraction_kept=self.fraction_kept).to(device) # h x w 

    def __call__(self, img):
        return self.A[None, ...] * img

    def __str__(self):
        return f'InpaintingScatter.fraction_kept={self.fraction_kept}'


# New code
class Denoising(ForwardModel):
    inverse = False

    def __init__(self, img_shape, sigma, device=DEFAULT_DEVICE):
        self.sigma = sigma
        self.noise = torch.normal(mean=0, std=sigma, size=img_shape).to(device)
        
    def __call__(self, img):
        return torch.clamp((img + self.noise), min=0, max=1)

    def __str__(self):
        return f'Denoising.sigma={self.sigma}'

    
class InpaintingIrregular(ForwardModel):
    """
    Mask rectangular pixels
    """
    inverse = True

    def __init__(self, img_shape, mask_dir, mask_name, device=DEFAULT_DEVICE):
        """
        img_shape - 3 x H x W
        fraction_kept - number in [0, 1], what portion of pixels to retain
        """    
        self.mask_name = mask_name
        mask = load_target_image(os.path.join(mask_dir, mask_name), img_shape[2]).to(device)
        mask[mask != 1.0] = 0.0 # fix to 1 and 0
        self.A = torch.abs(mask - 1.0)
        
    def __call__(self, img):
        return self.A[None, ...] * img
    
    def inverse(self, img):
        return (1.0 - self.A[None, ...]) * img

    def __str__(self):
        return f'InpaintingIrregular.name={self.mask_name}'

#

class SuperResolution(ForwardModel):
    inverse = False 

    def __init__(self, scale_factor, mode='linear', align_corners=True, **kwargs):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        res = F.interpolate(img, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return res

    def __str__(self):
        return (f'SuperResolution.scale_factor={self.scale_factor}' f'.mode={self.mode}')


def get_forward_model(fm_name, **fm_kwargs):
    return getattr(sys.modules[__name__], fm_name)(**fm_kwargs)

