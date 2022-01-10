#!/usr/bin/env bash

[[ $# -eq 3 ]] || { echo "Required args: model_name, img_dir, img_name, tv_weight" >&2; exit 1; }

MODEL=$1

# first cut = 3
# second cut = 15
python run_experiments.py \
    --img_dir $2 \
    --img_name $3 \
    --tv_weight 1e-8 \
    --model ${MODEL} \
