#!/usr/bin/env sh
# first cut = 3
# second cut = 15

# WORK_PATH=$(dirname $0)
# IMAGE_PATH=${1:-data/ILSVRC2012_val_00001970.JPEG}
# CLASS=${2:--1} 

python run_experiments.py \
--img_dir $1 \
--model began_inv \
--tv_weight 1e-8 \
--overwrite
