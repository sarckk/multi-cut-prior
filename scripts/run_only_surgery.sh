#!/usr/bin/env sh
# first cut = 3
# second cut = 15

# WORK_PATH=$(dirname $0)
# IMAGE_PATH=${1:-data/ILSVRC2012_val_00001970.JPEG}
# CLASS=${2:--1} 

for i in 0 1 3 ;     ### Outer for loop ###
    do
        echo "$i"
        python run_experiments.py --img_dir ./images/ood-examples --model began_inv --first_cut $i --tv_weight 1e-8 --overwrite 
    done
