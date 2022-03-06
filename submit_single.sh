#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -l gpu=1 
#$ -ac allow=EF
#$ -j y

IMG_LIST_FILE=$3
cp $IMG_LIST_FILE $TMPDIR
cp -R trained_model $TMPDIR

cd $TMPDIR

module unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
# Technically we don't have to load cuda as pytorch binaries come w/ cuda & cudnn
module load cuda/11.3.1/gnu-10.2.0
module load python3/3.8

# check gpu for debugging purposes
nvidia-smi 

source $HOME/multi-cut-prior/env/bin/activate

params=()
if [ "${10}" ]; then
    params+=(--optimizer ${10})
fi

if [ "${11}" ]; then
    params+=(--n_steps ${11})
fi

if [ "${12}" ]; then
    params+=(--restarts ${12})
fi

if [ "${13}" ]; then
    params+=(--mask_name ${13})
fi

/usr/bin/time --verbose python $HOME/multi-cut-prior/run_experiments.py \
--project_name $2 \
--model began \
--img_list $3 \
--img_dir $4 \
--base_dir $TMPDIR/output \
--mask_dir $5 \
--forward_model $6 \
--first_cut $7 \
--second_cut $8 \
--z_number $9 \
--disable_wandb \
--disable_tqdm \
"${params[@]}"

rsync -a --ignore-existing ./output/images/ ~/Scratch/merged-data/images/
rm -rf ./output/images

cd $TMPDIR/output
tar -cvf $HOME/Scratch/$1.tar.gz *
