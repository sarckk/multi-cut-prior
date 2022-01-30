import argparse
import os
import shutil
import warnings
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from utils import (get_z_vector, load_target_image, load_trained_net, psnr, psnr_from_mse)
import wandb

warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# From https://colab.research.google.com/github/rrmina/neural-style-pytorch/blob/master/neural_style_preserve_color.ipynb
def total_variation_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss


def get_opt(optimizer_type, params_dict, z_lr):
    params = list(params_dict.values())
    
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
    
    return optimizer_z, scheduler_z, save_img_every_n

def pack_losses(forward_model, x_hats_clamp, x, y_observed, y_masked_part):
    train_mse_clamped = F.mse_loss(forward_model(x_hats_clamp), y_observed)
    orig_mse_clamped = F.mse_loss(x_hats_clamp, x)

    if forward_model.inverse:
        masked_mse_clamped = F.mse_loss(forward_model.inverse(x_hats_clamp), y_masked_part)
    else:
        masked_mse_clamped = None

    orig_psnr = psnr_from_mse(orig_mse_clamped)

    loss_dict = {
        'TRAIN_MSE': train_mse_clamped,
        'ORIG_MSE': orig_mse_clamped,
        'ORIG_PSNR': orig_psnr,
    }

    if masked_mse_clamped is not None:
        loss_dict = {
            **loss_dict, 
            'ORIG_PSNR_ONLY_MASKED': psnr_from_mse(masked_mse_clamped)
        }
    
    return loss_dict

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
             log_every=1,
             **kwargs):

    uses_multicode = (second_cut != -1 and z_number > 1)
    
    z1_dim, z1_dim2 = gen.input_shapes[first_cut]
    
    num_codes = z_number if uses_multicode else 1 
    
    z1 = torch.nn.Parameter(get_z_vector((num_codes, *z1_dim), mode=mode, limit=limit, device=x.device))
    params_dict = {'z1': z1}
    
    if uses_multicode:
        alpha = torch.nn.Parameter(
            get_z_vector((z_number, gen.input_shapes[second_cut][0][0]), mode=mode, limit=limit, device=x.device))
        params_dict['alpha'] = alpha
    else:
        alpha = None

    if len(z1_dim2) > 0:
        z1_2 = torch.nn.Parameter(get_z_vector((num_codes, *z1_dim2), mode=mode, limit=limit, device=x.device))
        params_dict['z1_2'] = z1_2
    else:
        z1_2 = None
    
    if uses_multicode:
        _, z2_dim = gen.input_shapes[second_cut]
        if len(z2_dim) > 0:
            z2 = torch.nn.Parameter(get_z_vector((1, *z2_dim), mode=mode, limit=limit, device=x.device))
            params_dict['z2'] = z2
        else:
            z2 = None
            
    optimizer_z, scheduler_z, save_img_every = get_opt(optimizer_type, params_dict, z_lr)


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
                x_hats = gen.forward(F_l_2, z2, n_cuts=second_cut, end=-1, **kwargs)
                
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
                
            train_mse = F.mse_loss(forward_model(x_hats), y_observed)
            
            if tv_weight != 0.0:
                tv_loss = total_variation_loss(x_hats)
            else:
                tv_loss = 0.0
            
            loss = train_mse + tv_weight * tv_loss
            loss.backward()
            return loss

        
        optimizer_z.step(closure)
        
        
        with torch.no_grad():
            x_hats = gen.forward(z1, z1_2, n_cuts=first_cut, end=second_cut, **kwargs)
            if uses_multicode:
                F_l_2 = (x_hats * alpha[:, :, None, None]).sum(0, keepdim=True) / z_number
                x_hats = gen.forward(F_l_2, z2, n_cuts=second_cut, end=-1, **kwargs)
            
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
           
        
        x_hats_clamp = x_hats.detach().clamp(0, 1)        # 1 x 3 x h x w
        
        loss_dict = pack_losses(forward_model, x_hats_clamp, x, y_observed, y_masked_part)
        
        global_idx = restart_idx * n_steps + j + 1
            
        if j % log_every == 0 and disable_tqdm:
            print(f'\nRestart: {restart_idx}, Step: {j}')
            print('\t'.join(f'{k}:{v:.3f}' for k, v in loss_dict.items()))
        
        if writer is not None:
            for k,v in loss_dict.items():
                writer.add_scalar(k, v, global_idx)
            
            if forward_model.inverse:
                writer.add_image('ORIG_MASKED', y_masked_part.clamp(0,1).squeeze(), global_idx)
            
            if j % save_img_every == 0:
                writer.add_image('Recovered', x_hats_clamp.squeeze(), global_idx)
                
        if scheduler_z is not None:
            scheduler_z.step()

    if writer is not None:
        writer.add_image('Final', x_hats_clamp.squeeze(), restart_idx)

    return x_hats_clamp.squeeze(), y_observed.squeeze(), loss_dict, params_dict

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
            log_every=1,
            **kwargs):

    best_psnr = -float("inf")
    best_return_val = None
    
    writer = None
    
    if not disable_wandb:
        writer = SummaryWriter(logdir)

    # Save original and distorted image
    if writer is not None:
        writer.add_image("Original/Clamp", x.clamp(0, 1))
        writer.add_image("Distorted/Clamp", forward_model(x.unsqueeze(0).clamp(0, 1)).squeeze(0))

    for i in trange(restarts, desc='Restarts', leave=False, disable=disable_tqdm):
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
                              log_every=log_every,
                              **kwargs)
        
        train_mse = return_val[2]['TRAIN_MSE']
        p_train = psnr_from_mse(train_mse)
        if math.isnan(p_train):
            raise ValueError(f"\n Restart [{i}]: nan value of psnr found, train_mse_clamped is {train_mse}")
            
        if p_train > best_psnr:
            best_psnr = p_train
            best_return_val = return_val

    if writer is not None:
        writer.add_image('Best recovered', best_return_val[0])
    
    writer.close()
    return best_return_val

