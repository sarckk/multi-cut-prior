import argparse
import os
import shutil
import warnings
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import (get_z_vector, load_target_image, load_trained_net, psnr, psnr_from_mse)
import wandb

warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# From https://colab.research.google.com/github/rrmina/neural-style-pytorch/blob/master/neural_style_preserve_color.ipynb#scrollTo=MgSAy8vi-wD9
def total_variation_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss

def _recover(x,
             gen,
             optimizer_type,
             first_cut,
             second_cut,
             forward_model,
             writer=None,
             mode='clamped_normal',
             limit=1,
             z_number=-1,
             z_lr=0.5,
             n_steps=2000,
             restart_idx=0,
             disable_tqdm=False,
             tv_weight=0.0,
             disable_wandb=False,
             **kwargs):

    uses_multicode = (second_cut is not None and z_number != -1)
    
    print(first_cut)
    z1_dim, z1_dim2 = gen.input_shapes[first_cut]
    
    num_codes = z_number if uses_multicode else 1 
    
    z1 = torch.nn.Parameter(get_z_vector((num_codes, *z1_dim), mode=mode, limit=limit, device=x.device))
    params = [z1]
    
    if uses_multicode:
        alpha = torch.nn.Parameter(
            get_z_vector((z_number, gen.input_shapes[second_cut][0][0]), mode=mode, limit=limit, device=x.device))
        params.append(alpha)
    else:
        alpha = None

    if len(z1_dim2) > 0:
        z1_2 = torch.nn.Parameter(get_z_vector((num_codes, *z1_dim2), mode=mode, limit=limit, device=x.device))
        params.append(z1_2)
    else:
        z1_2 = None
    
    if uses_multicode:
        _, z2_dim = gen.input_shapes[second_cut]
        if len(z2_dim) > 0:
            z2 = torch.nn.Parameter(get_z_vector((1, *z2_dim), mode=mode, limit=limit, device=x.device))
            params.append(z2)
        else:
            z2 = None
        
    if optimizer_type == 'sgd':
        optimizer_z = torch.optim.SGD(params, lr=z_lr)
        scheduler_z = None
        save_img_every_n = 50
    elif optimizer_type == 'adam':
        optimizer_z = torch.optim.Adam(params, lr=z_lr)
        scheduler_z = None
        # scheduler_z = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer_z, n_steps, 0.05 * z_lr)
        save_img_every_n = 50
    elif optimizer_type == 'lbfgs':
        optimizer_z = torch.optim.LBFGS(params, lr=z_lr)
        scheduler_z = None
        save_img_every_n = 2
    elif optimizer_type == 'adamW':
        optimizer_z = torch.optim.AdamW(params,
                                        lr=z_lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=0)
        scheduler_z = None
        save_img_every_n = 50
    else:
        raise NotImplementedError()


    # Recover image under forward model
    x = x.expand(1, *x.shape)
    y_observed = forward_model(x)
    
    if forward_model.inverse:
        y_masked_part = forward_model.inverse(x)

    for j in trange(n_steps,
                    leave=False,
                    desc='Recovery',
                    disable=disable_tqdm):
        
        def closure():
            optimizer_z.zero_grad()
            x_hats = gen.forward(z1, z1_2, n_cuts=first_cut, end=second_cut, **kwargs)
            if uses_multicode:
                F_l_2 = (x_hats * alpha[:, :, None, None]).sum(0, keepdim=True) / z_number
                x_hats = gen.forward(F_l_2, z2, n_cuts=second_cut, end=None, **kwargs)
                
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
                
            train_mse = F.mse_loss(forward_model(x_hats), y_observed)
            
            if tv_weight != 0.0:
                tv_loss = tv_weight * total_variation_loss(x_hats)
            else:
                tv_loss = 0.0
            
         
            loss = train_mse + tv_loss
            loss.backward()
            return train_mse

        
        optimizer_z.step(closure)
        
        
        with torch.no_grad():
            x_hats = gen.forward(z1, z1_2, n_cuts=first_cut, end=second_cut, **kwargs)
            if uses_multicode:
                F_l_2 = (x_hats * alpha[:, :, None, None]).sum(0, keepdim=True) / z_number
                x_hats = gen.forward(F_l_2, z2, n_cuts=second_cut, end=None, **kwargs)
            
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
       
        train_mse_clamped = F.mse_loss(forward_model(x_hats.detach().clamp(0, 1)), y_observed)
        orig_mse_clamped = F.mse_loss(x_hats.detach().clamp(0, 1), x)
        
        if forward_model.inverse:
            masked_mse_clamped = F.mse_loss(forward_model.inverse(x_hats.detach().clamp(0,1)), y_masked_part)
        else:
            masked_mse_clamped = None
        
        global_idx = restart_idx * n_steps + j + 1
        if writer is not None:
            writer.add_scalar('TRAIN_MSE', train_mse_clamped, global_idx)
            writer.add_scalar('ORIG_MSE', orig_mse_clamped, global_idx)
            writer.add_scalar('ORIG_PSNR', psnr_from_mse(orig_mse_clamped), global_idx)
            
            if forward_model.inverse:
                # only true for square inpainting 
                writer.add_image('ORIG_MASKED', y_masked_part.clamp(0,1).squeeze(0), global_idx)
                writer.add_scalar('ORIG_PSNR_ONLY_MASKED', psnr_from_mse(masked_mse_clamped), global_idx)

            
            if j % save_img_every_n == 0:
                writer.add_image('Recovered',
                                 x_hats.clamp(0, 1).squeeze(0), global_idx)


        if scheduler_z is not None:
            scheduler_z.step()

    if writer is not None:
        writer.add_image('Final', x_hats.clamp(0, 1).squeeze(0), restart_idx)

    return x_hats.clamp(0,1).squeeze(0), forward_model(x)[0], train_mse_clamped, masked_mse_clamped


def recover(x,
            gen,
            optimizer_type,
            first_cut,
            second_cut,
            forward_model,
            mode='clamped_normal',
            limit=1,
            z_number=-1,
            z_lr=0.5,
            n_steps=2000,
            restarts=1,
            logdir=None,
            disable_tqdm=False,
            tv_weight=0.0,
            disable_wandb=False,
            **kwargs):

    best_psnr = -float("inf")
    best_return_val = None
    
    print("tensorboard logdir: ", logdir)
    writer = SummaryWriter(logdir)

    # Save original and distorted image
    if writer is not None:
        writer.add_image("Original/Clamp", x.clamp(0, 1))
        if forward_model.viewable:
            writer.add_image("Distorted/Clamp", forward_model(x.unsqueeze(0).clamp(0, 1)).squeeze(0))

    for i in trange(restarts,
                    desc='Restarts',
                    leave=False,
                    disable=disable_tqdm):
        return_val = _recover(x=x,
                              gen=gen,
                              optimizer_type=optimizer_type,
                              first_cut=first_cut,
                              second_cut=second_cut,
                              forward_model=forward_model,
                              writer=writer,
                              mode=mode,
                              limit=limit,
                              z_number=z_number,
                              z_lr=z_lr,
                              n_steps=n_steps,
                              restart_idx = i,
                              disable_tqdm=disable_tqdm,
                              tv_weight=tv_weight,
                              disable_wandb=disable_wandb,
                              **kwargs)

        p = psnr_from_mse(return_val[2])
        if math.isnan(p):
            print(f"\n Restart [{i}]: nan value of psnr found, train_mse_clamped is {return_val[2]}")
        # sometimes p is nan here -> why?
        if p > best_psnr:
            best_psnr = p
            best_return_val = return_val

            
    if writer is not None:
        writer.add_image('Best recovered', best_return_val[0])
    
    writer.close()
    return best_return_val

