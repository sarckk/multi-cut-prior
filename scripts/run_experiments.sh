#!/usr/bin/env sh

PROJECT_NAME=${1:-test123}               
FM=${2:-InpaintingIrregular}
C1=${3:-0}
C2=${4:--1}

python run_experiments.py \
--project_name $PROJECT_NAME \
--model began \
--forward_model $FM \
--img_list image_list.txt \
--first_cut $C1 \
--second_cut $C2 \
--z_number 20 \
--overwrite \
--n_steps 40 \
${5:+--mask_name $5}