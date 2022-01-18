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
import hydra
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import structural_similarity

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

BASE_DIR = './runs'

def reset_gen(model):
    if model in ['mgan_began_inv', 'began_inv']:
        state_dict = torch.load('./trained_model/gen_208000.pth', map_location=DEVICE)
        gen = load_pretrained_began_gen(state_dict)
        gen = gen.eval().to(DEVICE)
        img_size = 128
    elif model in ['mgan_dcgan_inv', 'dcgan_inv']:
        state_dict = torch.load('./trained_model/dcgan.pth', map_location=DEVICE)
        gen = load_pretrained_dcgan_gen(state_dict)
        gen = gen.eval().to(DEVICE)
        img_size = 64
    elif model in ['mgan_biggan_inv', 'biggan_inv']:
        gen = BigGanSkip().to(DEVICE)
        img_size = 128
    else:
        raise NotImplementedError()
    return gen, img_size

def restore(args, metadata, z_number, first_cut, second_cut = None):
    z_init_mode_list = metadata['z_init_mode']
    limit_list = metadata['limit']
    assert len(z_init_mode_list) == len(limit_list)
    del (metadata['z_init_mode'])
    del (metadata['limit'])
    
    data_split =  Path(args.img_dir).name
    gen, img_size = reset_gen(args.model)
    img_shape = (3, img_size, img_size)
    
    dataset_name = 'coco2017test'
    
    forwards = forward_models[args.model]
    img_dataset = ImgDataset(args.img_dir, img_size)
    img_dataloader = DataLoader(img_dataset, batch_size = args.batch_size, shuffle=False)
    
    for i, (f, f_args_list) in enumerate(
                            tqdm(forwards.items(),
                                 desc='Forwards',
                                 leave=False,
                                 disable=args.disable_tqdm)):
        for f_args in tqdm(f_args_list,
                           desc=f'{f} Args',
                           leave=False,
                           disable=args.disable_tqdm):

            for img_batch in tqdm(img_dataloader,
                 desc='Image batches',
                 leave=True,
                 disable=args.disable_tqdm):
                
                # img_batch is a tensor of shape (batch_size, 3, img_size, img_size)
                img_batch = img_batch.to(DEVICE)
                               
                f_args['img_shape'] = img_shape
                forward_model = get_forward_model(f, **f_args)
                del (f_args['img_shape'])

                for z_init_mode, limit in zip(
                        tqdm(z_init_mode_list,
                             desc='z_init_mode',
                             leave=False), limit_list):
                    metadata['z_init_mode'] = z_init_mode
                    metadata['limit'] = limit
                    metadata['tv_weight'] = args.tv_weight

                    # Before doing recovery, check if results already exist and possibly skip
                    metadata_str = dict_to_str(metadata, exclude="img")
                    recovered_name = 'recovered.pt'
                    results_folder = get_results_folder(
                        dataset_name = dataset_name,
                        model=args.model,
                        n_cuts=first_cut if second_cut is None else str([first_cut, second_cut]),
                        split=data_split,
                        forward_model=forward_model,
                        recovery_params=metadata_str,
                        base_dir=BASE_DIR)

                    os.makedirs(results_folder, exist_ok=True)

                    recovered_path = results_folder / recovered_name
                    if os.path.exists(recovered_path) and not args.overwrite:
                        print(f'{recovered_path} already exists, skipping...')
                        continue

                    current_run_name = (
                        f'coco2017.{forward_model}'
                        f'.{metadata_str}')

                    if args.run_name is not None:
                        current_run_name = current_run_name + f'.{args.run_name}'

                    run_dir = args.model
                    logdir = os.path.join('recovery_tensorboard_logs', run_dir, current_run_name)
                    if os.path.exists(logdir):
                        print("Overwriting pre-existing logs!")
                        shutil.rmtree(logdir)

                    # wandb logging       
                    if not args.disable_wandb:
                        # wandb.tensorboard.patch(root_logdir=logdir)
                        wandb_run = wandb.init(
                            project="BATCH_IMAGES", 
                            group=f + ', ' + dict_to_str(f_args),
                            name=current_run_name, 
                            tags=[args.model, data_split, dataset_name, f],
                            config=metadata, 
                            reinit=True,
                            save_code=True,
                            sync_tensorboard=True
                        )

                    start = time.time()
                    recovered_img, distorted_img, _, masked_mse = recover(
                        img_batch, gen, metadata['optimizer'], 
                        first_cut,
                        second_cut, 
                        forward_model, z_init_mode, limit, 
                        z_number,
                        metadata['z_lr'], metadata['n_steps'],
                        metadata['restarts'], logdir, args.disable_tqdm, 
                        args.tv_weight, args.disable_wandb
                    )
                    time_taken = time.time() - start

                    p = psnr(recovered_img.cpu().numpy(), img_batch.cpu().numpy())
                    wandb.run.summary['best_origin_psnr'] = p
                    ssim = structural_similarity(recovered_img.cpu().numpy(), img_batch.cpu().numpy(), channel_axis=1, data_range=1.0)
                    wandb.run.summary['best_origin_ssim'] = ssim
                    wandb.run.summary['time_taken'] = time_taken
                    wandb.run.summary['batch_size'] = args.batch_size 

                    if not args.disable_wandb:
                        wandb_run.finish()

                    # Make images folder
                    img_folder = get_images_folder(split=data_split,
                                                   image_name=dataset_name,
                                                   img_size=img_size,
                                                   base_dir=BASE_DIR)
                    os.makedirs(img_folder, exist_ok=True)

                    # Save original image if needed
                    original_img_path = img_folder / 'original.pt'
                    if not os.path.exists(original_img_path):
                        torch.save(torchvision.utils.make_grid(img_batch,nrow), original_img_path)

                    # Save distorted image if needed
                    if forward_model.viewable:
                        distorted_img_path = img_folder / f'{forward_model}.pt'
                        if not os.path.exists(distorted_img_path):
                            torch.save(make_grid(distorted_img), distorted_img_path)

                    # Save recovered image and metadata
                    torch.save(make_grid(recovered_img), recovered_path) # results_folder/ recovered.pt

                    pickle.dump(metadata, open(results_folder / 'metadata.pkl', 'wb'))
                    pickle.dump(p, open(results_folder / 'psnr.pkl', 'wb'))
                    pickle.dump(ssim, open(results_folder / 'ssim.pkl', 'wb'))

                    if masked_mse is not None:
                        # only here if inpainting square 
                        p_masked = psnr_from_mse(masked_mse)
                        pickle.dump(p_masked, open(results_folder / 'psnr_masked.pkl', 'wb'))


def gan_images(args, metadata):
    # extra processing of metadata
#     n_cuts_list = metadata['n_cuts_list']
#     del (metadata['n_cuts_list'])
    
    first_cut = args.first_cut
    metadata['cut'] = f'{first_cut},-1'
    
    print(f"===> Testing out combination: [{metadata['cut']}]")

    restore(args, metadata, -1, first_cut, second_cut=None)
        
        
def mgan_images(args, metadata):
    # extra processing of metadata
    first_cut = args.first_cut
    second_cut = args.second_cut
    metadata['cut'] = f'{first_cut},{second_cut}'
    print(f"===> Testing out combination: [{metadata['cut']}]")
    
    z_number = metadata['z_number']
    
    restore(args, metadata, z_number, first_cut, second_cut=second_cut)

    
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    
    # mgan
    p.add_argument('--img_dir', required=True, help='')
#     p.add_argument('--img_name', required=True, help='')
    p.add_argument('--first_cut', default=None, type=int)
    p.add_argument('--second_cut', default=None, type=int)
    
    # training
    p.add_argument('--tv_weight', type=float, default=1e-8)
    p.add_argument('-b', '--batch_size', type=int, default=1)
    
    # logging
    p.add_argument('--disable_wandb', help='Disable weights and biases logging', action='store_true', default=False)
    
    p.add_argument('--run_name', default=None)
    p.add_argument('--disable_tqdm', action='store_true')
    p.add_argument('--overwrite',
                   action='store_true',
                   help='Set flag to overwrite pre-existing files')

    
    args = p.parse_args()
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    metadata = recovery_settings[args.model]
    metadata['tv_weight'] = args.tv_weight

    if args.model in [
            'began_inv',
            'biggan_inv',
            'dcgan_inv',
    ]:
        gan_images(args, metadata)
    elif args.model in [
            'mgan_began_inv',
            'mgan_dcgan_inv',
            'mgan_biggan_inv',
    ]:
        mgan_images(args, metadata)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
