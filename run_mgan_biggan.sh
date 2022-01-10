#!/bin/bash
for (( i = 1; i <= 13; i++ ))      ### Outer for loop ###
do

    for (( j = 0; j < i; j++ )) ### Inner for loop ###
    do
          echo -n "first_cut: $j, second_cut: $i"
	  python run_experiments.py --img_dir ./images/ood-examples/ --img_name $1.jpg  --model mgan_biggan_inv --run_dir mgan_biggan_inv --run_name mgan_biggan_inv_norestart --first_cut $j --second_cut $i --overwrite
    done

  echo "" #### print the new line ###
done

