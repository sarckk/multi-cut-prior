import argparse
import os
import pickle
from pathlib import Path
import logging
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
from settings import forward_models
from utils import (dict_to_str, get_images_folder,
                   get_results_folder, load_target_image, load_trained_net,
                   psnr, psnr_from_mse, load_pretrained_dcgan_gen, load_pretrained_began_gen, ImgDataset,
                  setup_logger, get_logs_folder, ROOT_LOGGER_NAME)
import wandb
from skimage.metrics import structural_similarity as calc_ssim


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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



def verify_args(args):
    if args.model not in ['began', 'biggan', 'dcgan']:
        raise NotImplementedError()
    
    if args.project_name is None and not args.disable_wandb:
        raise ValueError('Project name is required if you enable wandb logging')
    
    if not os.path.isdir(args.mask_dir):
        raise ValueError('Mask directory is invalid')
    
    if args.second_cut == 0:
        raise ValueError('Argument second_cut can either be -1 or >0')
    
    if args.second_cut != -1 and args.second_cut <= args.first_cut:
        raise ValueError('Argument second_cut must be greater than args.first_cut')
    
    if args.second_cut != -1 and args.z_number <= 1:
        raise ValueError('For mGANPrior, use multiple latent codes. Otherwise there is no difference!')
    

    
def restore(args, metadata, z_number, first_cut, second_cut):
    dataset_name =  Path(args.img_dir).name
    gen, img_size = reset_gen(args.model)
    img_shape = (3, img_size, img_size)
    
    f_args = forward_models[args.forward_model]
    
    logger = setup_logger(ROOT_LOGGER_NAME, get_logs_folder(args.base_dir, args.project_name))
    
    # try overriding mask name in f_args if forward model is IrregularInpainting
    if args.forward_model == 'InpaintingIrregular' and args.mask_name is not None:
        assert 'mask_name' in f_args.keys()
        print(f'===> Overriding mask_name to {args.mask_name}')
        f_args['mask_name'] = args.mask_name 
        
    
    img_dataset = ImgDataset(args.img_dir, args.img_list, img_size)
    img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False)
 
    cuts_combination = str([first_cut, second_cut])
    
    for idx, (image, img_name) in enumerate(tqdm(img_dataloader, desc='Images', leave=True, disable=args.disable_tqdm)):
        image = image.squeeze().to(DEVICE)  # remove batch dimension 
        img_name = img_name[0]
        img_basename, _ = os.path.splitext(img_name)

        f_args['img_shape'] = img_shape

        if args.forward_model == 'InpaintingIrregular':
            f_args['mask_dir'] = args.mask_dir
            
        forward_model = get_forward_model(args.forward_model, **f_args)
        del (f_args['img_shape'])
        del (f_args['mask_dir'])

        metadata['img'] = img_basename
        metadata_str = dict_to_str(metadata, exclude=['img', 'cut'])

        results_folder = get_results_folder(
            image_name=img_basename,
            model=args.model,
            cuts=cuts_combination,
            dataset=dataset_name,
            forward_model=forward_model,
            recovery_params=metadata_str,
            base_dir=args.base_dir
        )

        os.makedirs(results_folder, exist_ok=True)

        recovered_path = results_folder / 'recovered.pt'
        if os.path.exists(recovered_path) and not args.overwrite:
            print(f'{recovered_path} already exists, skipping...')
            continue

        current_run_name = (
            f'{img_basename}.{forward_model}'
            f'.{metadata_str}.cut={metadata["cut"]}'
        )

        if args.run_name is not None:
            current_run_name = current_run_name + f'.{args.run_name}'


        logdir = os.path.join('recovery_tensorboard_logs', args.model, current_run_name)
        if os.path.exists(logdir):
            print("Overwriting pre-existing logs!")
            shutil.rmtree(logdir)


        if not args.disable_wandb:
            wandb_run = wandb.init(
                project= args.project_name, 
                group= args.forward_model + ', ' + dict_to_str(f_args),
                name=current_run_name, 
                tags=[args.model, dataset_name, args.forward_model],
                config=metadata, 
                reinit=True,
                save_code=True,
                sync_tensorboard=True
            )

        start = time.time()
        recovered_img, distorted_img, loss_dict, best_params, is_valid = recover(
            image, 
            gen, 
            metadata['optimizer'], 
            first_cut,
            second_cut, 
            forward_model, 
            args.z_init_mode, 
            args.limit, 
            metadata['z_number'],
            metadata['z_lr'], 
            metadata['n_steps'],
            metadata['restarts'], 
            logdir, 
            args.disable_tqdm, 
            metadata['tv_weight'], 
            args.disable_wandb, 
            args.print_every
        )
        time_taken = time.time() - start
        
        # this prints to stdout
        logger.info(f'[{idx+1}/{len(img_dataloader)}]  Img: {img_basename}  Time taken: {time_taken:.3f}')
        
        # this saves to log destination file
        if not is_valid:
            logger.warning(f'[{idx+1}/{len(img_dataloader)}]  Run: {current_run_name}  Potentially invalid run.')
        
        
        p = loss_dict['ORIG_PSNR']
        ssim = calc_ssim(recovered_img.cpu().numpy(), image.cpu().numpy(), channel_axis=0, data_range=1.0)
        metrics = {k: v for k,v in loss_dict.items() if 'PSNR' in k}
        metrics['ORIG_SSIM'] = ssim
        metrics['time_taken'] = time_taken

        if not args.disable_wandb:
            wandb.run.summary['best_origin_psnr'] = p
            wandb.run.summary['best_origin_ssim'] = ssim
            if forward_model.inverse:
                wandb.run.summary['best_masked_psnr'] = loss_dict['ORIG_PSNR_ONLY_MASKED']

            wandb.run.summary['time_taken'] = time_taken
            wandb_run.finish()


        # Make images folder
        img_folder = get_images_folder(dataset=dataset_name,
                                       image_name=img_basename,
                                       img_size=img_size,
                                       base_dir=args.base_dir)
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
    
    if second_cut == -1:
        z_number = 1 # GS can only use 1 latent code
        print(f"===> Testing Generator Surgery")
    else:
        if first_cut == 0:
            print(f"===> Testing Multi-code GAN Prior")
        else:
            print(f"===> Testing our method")
    
    cut_str = f'{first_cut},{second_cut}'
    print(f"===> Using {z_number} latent codes")
    print(f"===> Testing out combination: [{cut_str}]")
    
    metadata['z_number'] = z_number
    metadata['cut'] = cut_str

    restore(args, metadata, z_number, first_cut=first_cut, second_cut=second_cut)


def main():
    p = argparse.ArgumentParser()
    
    # core
    p.add_argument('--model', required=True)
    p.add_argument('--forward_model', required=True, choices=['InpaintingIrregular', 'InpaintingScatter', 'SuperResolution'])
    p.add_argument('--img_dir', default='./images/test2017')
    p.add_argument('--base_dir', default='./logs')
    p.add_argument('--img_list', required=True)
    p.add_argument('--first_cut', default=0, type=int)
    p.add_argument('--second_cut', default=-1, type=int)
    p.add_argument('-z', '--z_number', default=20, type=int)
    
    # mask related overrides
    p.add_argument('--mask_name', default=None)                                 # only overrides for irregular inpainting mask
    p.add_argument('--mask_dir', default='./images/mask/testing_mask_dataset') 
    
    # training
    p.add_argument('--tv_weight', type=float, default=1e-8)
    p.add_argument('--restarts', type=int, default=1)
    p.add_argument('--n_steps', type=int, default=40)
    p.add_argument('--z_lr', type=float, default=1.0)
    p.add_argument('--limit', default=1, type=int)
    p.add_argument('--optimizer', default='lbfgs', choices=['lbfgs', 'adam', 'adamW', 'sgd'])
    p.add_argument('--z_init_mode', default='clamped_normal', choices=['clamped_normal', 'normal', 'truncated_normal', 'rectified_normal', 'uniform', 'zero'])
     
    # run-related 
    p.add_argument('--disable_wandb', help='Disable weights and biases logging', action='store_true')
    p.add_argument('--run_name', default=None)
    p.add_argument('--project_name', required=True)
    p.add_argument('--print_every', default=10, type=int)
    p.add_argument('--disable_tqdm', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='Set flag to overwrite pre-existing files')

    args = p.parse_args()
    
    verify_args(args)
    
    os.makedirs(args.base_dir, exist_ok=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    metadata = dict()
    metadata['tv_weight'] = args.tv_weight
    metadata['optimizer'] = args.optimizer
    metadata['n_steps'] = args.n_steps
    metadata['z_lr'] = args.z_lr
    metadata['restarts'] = args.restarts


    gan_images(args, metadata)

if __name__ == '__main__':
    main()
