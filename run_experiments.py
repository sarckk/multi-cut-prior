import argparse
import os
import pickle
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from forward_model import get_forward_model
from model.biggan import BigGanSkip
from model.dcgan import Generator as dcgan_generator
from recover import recover
from settings import forward_models, recovery_settings
from utils import (dict_to_str, get_images_folder,
                   get_results_folder, load_target_image, load_trained_net,
                   psnr, psnr_from_mse, load_pretrained_dcgan_gen, load_pretrained_began_gen, ImgDataset)
import wandb
from skimage.metrics import structural_similarity as calc_ssim

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_DIR = './logs'


def reset_gen(model):
    if model == 'began':
        state_dict = torch.load('./trained_model/gen_208000.pth', map_location=DEVICE)
        gen = load_pretrained_began_gen(state_dict)
        gen = gen.eval().to(DEVICE)
        img_size = 128
    elif model == 'dcgan':
        state_dict = torch.load('./trained_model/dcgan.pth', map_location=DEVICE)
        gen = load_pretrained_dcgan_gen(state_dict)
        gen = gen.eval().to(DEVICE)
        img_size = 64
    elif model == 'biggan':
        gen = BigGanSkip().to(DEVICE)
        img_size = 128
    else:
        raise NotImplementedError()
    return gen, img_size


def restore(args, metadata, z_number, first_cut, second_cut=None):
    data_split =  Path(args.img_dir).name
    gen, img_size = reset_gen(args.model)
    img_shape = (3, img_size, img_size)
    
    forwards = forward_models[args.model]
    img_dataset = ImgDataset(args.img_dir, args.img_list, img_size)
    img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False)
 
    cuts_combination = first_cut if second_cut is None else str([first_cut, second_cut])
    
    for i, (f, f_args_list) in enumerate(
                            tqdm(forwards.items(),
                                 desc='Forwards',
                                 leave=False,
                                 disable=args.disable_tqdm)):
        for f_args in tqdm(f_args_list,
                           desc=f'{f} Args',
                           leave=False,
                           disable=args.disable_tqdm):

            for image, img_name in tqdm(img_dataloader, desc='Images', leave=True, disable=args.disable_tqdm):
                image = image.squeeze().to(DEVICE)  # remove batch dimension 
                img_name = img_name[0]
                img_basename, _ = os.path.splitext(img_name)

                f_args['img_shape'] = img_shape
                forward_model = get_forward_model(f, **f_args)
                del (f_args['img_shape'])
            
                metadata['img'] = img_basename
                metadata_str = dict_to_str(metadata, exclude='img')

                # Before doing recovery, check if results already exist and possibly skip
                metadata_str = dict_to_str(metadata, exclude="img")
                
                results_folder = get_results_folder(
                    image_name=img_basename,
                    model=args.model,
                    cuts=cuts_combination,
                    split=data_split,
                    forward_model=forward_model,
                    recovery_params=metadata_str,
                    base_dir=BASE_DIR
                )

                os.makedirs(results_folder, exist_ok=True)

                recovered_path = results_folder / 'recovered.pt'
                if os.path.exists(recovered_path) and not args.overwrite:
                    print(f'{recovered_path} already exists, skipping...')
                    continue

                current_run_name = (
                    f'{img_basename}.{forward_model}'
                    f'.{metadata_str}'
                )

                if args.run_name is not None:
                    current_run_name = current_run_name + f'.{args.run_name}'

                    
                logdir = os.path.join('recovery_tensorboard_logs', args.model, current_run_name)
                if os.path.exists(logdir):
                    print("Overwriting pre-existing logs!")
                    shutil.rmtree(logdir)

                    
                if not args.disable_wandb:
                    wandb_run = wandb.init(
                        project="lmao", 
                        group=f + ', ' + dict_to_str(f_args),
                        name=current_run_name, 
                        tags=[args.model, data_split, "coco2017", f],
                        config=metadata, 
                        reinit=True,
                        save_code=True,
                        sync_tensorboard=True
                    )

                start = time.time()
                recovered_img, distorted_img, loss_dict, best_params = recover(
                    image, 
                    gen, 
                    metadata['optimizer'], 
                    first_cut,
                    second_cut, 
                    forward_model, 
                    metadata['z_init_mode'], 
                    metadata['limit'], 
                    z_number,
                    metadata['z_lr'], 
                    metadata['n_steps'],
                    metadata['restarts'], 
                    logdir, 
                    args.disable_tqdm, 
                    args.tv_weight, 
                    args.disable_wandb, 
                    args.log_every
                )
                time_taken = time.time() - start
                
                p = loss_dict['ORIG_PSNR']
                masked_psnr = loss_dict['ORIG_PSNR_ONLY_MASKED']
                ssim = calc_ssim(recovered_img.cpu().numpy(), image.cpu().numpy(), channel_axis=0, data_range=1.0)
                metrics = {k: v for k,v in loss_dict.items() if 'PSNR' in k}
                metrics['ORIG_SSIM'] = ssim
                
                if not args.disable_wandb:
                    wandb.run.summary['best_origin_psnr'] = p
                    wandb.run.summary['best_origin_ssim'] = ssim
                    if forward_model.inverse:
                        wandb.run.summary['best_masked_psnr'] = masked_psnr
                        
                    wandb.run.summary['time_taken'] = time_taken
                    wandb_run.finish()

         
                # Make images folder
                img_folder = get_images_folder(split=data_split,
                                               image_name=img_basename,
                                               img_size=img_size,
                                               base_dir=BASE_DIR)
                os.makedirs(img_folder, exist_ok=True)

                # Save original image if needed
                original_img_path = img_folder / 'original.pt'
                if not os.path.exists(original_img_path):
                    torch.save(image, original_img_path)

                # Save distorted image if needed
                distorted_img_path = img_folder / f'{forward_model}.pt'
                if not os.path.exists(distorted_img_path):
                    torch.save(distorted_img, distorted_img_path)

                # Save recovered image and metadata
                torch.save(recovered_img, recovered_path) # results_folder/ recovered.pt
                pickle.dump(metrics, open(results_folder / 'metrics.pkl', 'wb'))
                pickle.dump(best_params, open(results_folder / 'best_params.pkl', 'wb'))


def gan_images(args, metadata):
    first_cut = args.first_cut
    second_cut = args.second_cut
    z_number = args.z_number # this doesn't matter for GS
    
    if second_cut is not None and z_number <= 1:
        raise ValueError('For mGANPrior, use multiple latent codes. Otherwise there is no difference!')
    
    if second_cut is None:
        metadata['cut'] = f'{first_cut},-1'
        z_number = 1 # GS can only use 1 latent code
        print(f"===> Testing Generator Surgery")
    else:
        metadata['cut'] = f'{first_cut},{second_cut}'
        if first_cut == 0:
            print(f"===> Testing Multi-code GAN Prior")
        else:
            print(f"===> Testing our method")

    print(f"==> Using {z_number} latent codes")
    print(f"===> Testing out combination: [{metadata['cut']}]")

    restore(args, metadata, z_number, first_cut=first_cut, second_cut=second_cut)

    
def main():
    p = argparse.ArgumentParser()
    
    #core
    p.add_argument('--model', required=True)
    p.add_argument('--img_dir', required=True, help='')
    p.add_argument('--img_list', required=True)
    p.add_argument('--first_cut', default=None, type=int)
    p.add_argument('--second_cut', default=None, type=int)
    p.add_argument('-z', '--z_number', default=20, type=int)
    
    # training
    p.add_argument('--tv_weight', type=float, default=1e-8)
    p.add_argument('--limit', default=1, type=int)
    p.add_argument('--z_init_mode', default='clamped_normal', choices=['clamped_normal', 'normal', 'truncated_normal', 'rectified_normal', 'uniform', 'zero'])
    
    # logging
    p.add_argument('--disable_wandb', help='Disable weights and biases logging', action='store_true')
    
    # run-related 
    p.add_argument('--run_name', default=None)
    p.add_argument('--log_every', default=5, type=int)
    p.add_argument('--disable_tqdm', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='Set flag to overwrite pre-existing files')

    args = p.parse_args()
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    metadata = recovery_settings[args.model]
    metadata['tv_weight'] = args.tv_weight
    metadata['z_init_mode'] = args.z_init_mode
    metadata['limit'] = args.limit
    
    if args.model not in ['began', 'biggan', 'dcgan']:
        raise NotImplementedError()

    gan_images(args, metadata)

if __name__ == '__main__':
    main()
