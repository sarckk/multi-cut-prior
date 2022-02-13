#!/usr/bin/env sh

PROJECT_NAME=${1:-test123}               
FM=${2:-InpaintingIrregular}
C1=${3:-0}
C2=${4:--1}
C3=${5:-0}

python run_experiments.py \
--project_name $PROJECT_NAME \
--model began \
--forward_model $FM \
--img_list imgnet.txt \
--base_dir ./todelete_logs \
--first_cut $C1 \
--second_cut $C2 \
--z_number 20 \
--overwrite \
--n_steps 30 \
--restarts 1 \
--cos_weight $C3 \
--disable_tqdm \
--disable_wandb \
${6:+--mask_name $6}