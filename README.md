# Multi-cut GAN Prior

<img width="1335" alt="Screenshot 2022-09-03 at 20 26 37" src="https://user-images.githubusercontent.com/48474650/188285225-c430086e-7f7d-4bf0-af22-2b763b9b42cf.png">

> With increased accessibility to pre-trained GAN models, many techniques have been proposed to exploit the rich landscape of information captured by GANs for image restoration. However, it still remains a challenging task due to the potential mismatch between the distribution of training data of the GAN and the target data. Some works propose fine-tuning the GAN weights, but this is costly and may not always be viable. Other methods, such as Generator Surgery (GS) and Multi-code GAN Prior (mGANPrior) iteratively update the input to the GAN at various layers while leaving the GAN weights intact. However, we find that these methods yield poor results on more challenging image restoration tasks such as inpainting and super-resolution when there is a large mismatch in data distribution. In this work, we present Multi-cut GAN Prior (MGP), a technique that combines GS and mGANPrior into a generalized formulation, and evaluate it in this data-mismatch regime. Our results show that MGP outperforms both GS and mGANPrior for various image restoration tasks such as denoising, inpainting and super-resolution.

## Quick start
First install all dependencies with `pip`:

```python
pip install -r requirements.txt
```

Then you can run `run_experiments.py` to run the actual experiments, with appropriate commands:

```bash
usage: run_experiments.py [-h] --model {began} --forward_model {InpaintingIrregular,InpaintingScatter,SuperResolution,Denoising} [--img_dir IMG_DIR] [--base_dir BASE_DIR] --img_list IMG_LIST
                          [--first_cut FIRST_CUT] [--second_cut SECOND_CUT] [-z Z_NUMBER] [--mask_name MASK_NAME] [--mask_dir MASK_DIR] [--tv_weight TV_WEIGHT] [--restarts RESTARTS] [--n_steps N_STEPS]
                          [--z_lr Z_LR] [--limit LIMIT] [--optimizer {lbfgs,adam,adamW,sgd}] [--z_init_mode {clamped_normal,normal,truncated_normal,rectified_normal,uniform,zero}] [--disable_wandb]
                          [--save_params] [--disable_tqdm] [--run_name RUN_NAME] --project_name PROJECT_NAME [--print_every PRINT_EVERY] [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  --model {began}
  --forward_model {InpaintingIrregular,InpaintingScatter,SuperResolution,Denoising}
  --img_dir IMG_DIR
  --base_dir BASE_DIR
  --img_list IMG_LIST
  --first_cut FIRST_CUT
  --second_cut SECOND_CUT
  -z Z_NUMBER, --z_number Z_NUMBER
  --mask_name MASK_NAME
  --mask_dir MASK_DIR
  --tv_weight TV_WEIGHT
  --restarts RESTARTS
  --n_steps N_STEPS
  --z_lr Z_LR
  --limit LIMIT
  --optimizer {lbfgs,adam,adamW,sgd}
  --z_init_mode {clamped_normal,normal,truncated_normal,rectified_normal,uniform,zero}
  --disable_wandb       Disable weights and biases logging
  --save_params
  --disable_tqdm
  --run_name RUN_NAME
  --project_name PROJECT_NAME
  --print_every PRINT_EVERY
  --overwrite           Set flag to overwrite pre-existing files

```

## Running on cluster
Experiments were run on a HPC cluster. The script for submitting a single job is given in `submit_single.sh`:

## Acknowledgements
The code in this repo was forked from https://github.com/nik-sm/generator-surgery (Generator Surgery for Compressed Sensing
by Jung Yeon Park\*, Niklas Smedemark-Margulies\*, Max Daniels, Rose Yu, Jan-Willem van de Meent, and Paul Hand) and later modified.
Refer to the git commit history for the exact changes.

Official implementation of LPIPS was used: https://github.com/richzhang/PerceptualSimilarity

Pretrained weights for BEGAN comes from https://github.com/zhusiling/BEGAN/tree/master/trained_models/128_tanh/models

We used the ImageNet mini dataset from [here](https://www.kaggle.com/ifigotin/imagenetmini-1000) for validation.
