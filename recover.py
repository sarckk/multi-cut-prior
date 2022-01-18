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

# From https://colab.research.google.com/github/rrmina/neural-style-pytorch/blob/master/neural_style_preserve_color.ipynb#scrollTo=MgSAy8vi-wD9
def total_variation_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss


def calc_batch_average(feats, alpha, nc, bs):
    F_l_2_list = []
    for k in range(bs):
        F_l_2 = (feats[k*nc:k*nc+nc] * alpha[k*nc:k*nc+nc, :, None, None]).sum(0,keepdim=True) # nc x h x w 
        F_l_2_list.append(F_l_2 / nc)

    return torch.cat(F_l_2_list,dim=0) 

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
    bs = x.size(0) # batch size
    
    z1_dim, z1_dim2 = gen.input_shapes[first_cut]
    
    nc = z_number if uses_multicode else 1 
    
    z1 = torch.nn.Parameter(get_z_vector((bs * nc, *z1_dim), mode=mode, limit=limit, device=x.device))
    params = [z1]
    
    if uses_multicode:
        alpha = torch.nn.Parameter(
            get_z_vector((bs * nc, gen.input_shapes[second_cut][0][0]), mode=mode, limit=limit, device=x.device))
        params.append(alpha)
    else:
        alpha = None

    if len(z1_dim2) > 0:
        z1_2 = torch.nn.Parameter(get_z_vector((bs * nc, *z1_dim2), mode=mode, limit=limit, device=x.device))
        params.append(z1_2)
    else:
        z1_2 = None
    
    if uses_multicode:
        _, z2_dim = gen.input_shapes[second_cut]
        if len(z2_dim) > 0:
            z2 = torch.nn.Parameter(get_z_vector((bs, *z2_dim), mode=mode, limit=limit, device=x.device))
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
    y_observed = forward_model(x)
    print("degraded observation shape: ", y_observed.shape)
    # y_observed is of dimensions bs x 3 x h x w
    
    if forward_model.inverse:
        y_masked_part = forward_model.inverse(x)

    for j in trange(n_steps,
                    leave=False,
                    desc='Recovery',
                    disable=disable_tqdm):
        
        def closure():
            optimizer_z.zero_grad()
            x_hats = gen.forward(z1, z1_2, n_cuts=first_cut, end=second_cut, **kwargs) 
            
            # x_hats is of shape (bs x nc) x 3 x h' x w'
            # for non-multicode (normal BEGAN w/ or w/o GS), nc=1 so x_hats is bs x c x h' x w'
            
            
            if uses_multicode:
                F_l_2_batched = calc_batch_average(x_hats, alpha, nc, bs) # bs x c x h' x w'
                # z2 is also shape bs x c x h' x w'
                x_hats = gen.forward(F_l_2_batched, z2, n_cuts=second_cut, end=None, **kwargs) # bs x 3 x h x w 
            
            
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
                
            train_mse = F.mse_loss(forward_model(x_hats), y_observed)
            
            if tv_weight != 0.0:
                tv_loss = tv_weight * total_variation_loss(x_hats)
            else:
                tv_loss = 0.0
            
         
            loss = train_mse + tv_loss
            loss.backward()
            return loss

        
        optimizer_z.step(closure)
        
        
        with torch.no_grad():
            x_hats = gen.forward(z1, z1_2, n_cuts=first_cut, end=second_cut, **kwargs)
            if uses_multicode:
                F_l_2 = calc_batch_average(x_hats, alpha, nc, bs)
                x_hats = gen.forward(F_l_2, z2, n_cuts=second_cut, end=None, **kwargs)
            
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
           
        
        print("final image shape: ", x_hats.shape)
        x_hats_clamp = x_hats.detach().clamp(0, 1) # bs x 3 x h x w
        train_mse_clamped = F.mse_loss(forward_model(x_hats_clamp), y_observed)
        orig_mse_clamped = F.mse_loss(x_hats_clamp, x)
        
        if forward_model.inverse:
            masked_mse_clamped = F.mse_loss(forward_model.inverse(x_hats_clamp), y_masked_part)
        else:
            masked_mse_clamped = None
        
        global_idx = restart_idx * n_steps + j + 1
        if writer is not None:
            writer.add_scalar('TRAIN_MSE', train_mse_clamped, global_idx)
            writer.add_scalar('ORIG_MSE', orig_mse_clamped, global_idx)
            writer.add_scalar('ORIG_PSNR', psnr_from_mse(orig_mse_clamped), global_idx)
            
            if forward_model.inverse:
                # only true for square inpainting 
                writer.add_image('ORIG_MASKED', make_grid(y_masked_part.clamp(0,1)), global_idx)
                writer.add_scalar('ORIG_PSNR_ONLY_MASKED', psnr_from_mse(masked_mse_clamped), global_idx)
            
            if j % save_img_every_n == 0:
                writer.add_image('Recovered', make_grid(x_hats_clamp), global_idx)

        if scheduler_z is not None:
            scheduler_z.step()

    if writer is not None:
        writer.add_image('Final', make_grid(x_hats_clamp), restart_idx)

    return x_hats_clamp, y_observed, train_mse_clamped, masked_mse_clamped


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
    
    writer = SummaryWriter(logdir)

    # Save original and distorted image
    if writer is not None:
        writer.add_image("Original/Clamp", make_grid(x.clamp(0, 1)))
        if forward_model.viewable:
            writer.add_image("Distorted/Clamp", make_grid(forward_model(x.clamp(0, 1))))

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
        writer.add_image('Best recovered', make_grid(best_return_val))
        if return_val[3] is not None:
            # there is a masked inverse part
            wandb.run.summary['best_masked_psnr'] = psnr_from_mse(best_return_val[3])
    
    writer.close()
    return best_return_val



if __name__ == '__main__':
    print("testing...")
    # save images of inverse 
    
    # check 