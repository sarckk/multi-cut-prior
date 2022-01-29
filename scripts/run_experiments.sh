#!/usr/bin/env sh

IMG_LIST=${1:-./image_list.txt}


echo $IMG_LIST

python run_experiments.py \
--img_dir ./images/test2017 \
--img_list $IMG_LIST \
--model began \
--z_number 20 \
--first_cut 3 \
--second_cut 15 \
--tv_weight 1e-8 \
--log_every 10 \
--overwrite \
