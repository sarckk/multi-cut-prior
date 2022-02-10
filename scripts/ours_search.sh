#!/usr/bin/bash

for((first=1;first<=5;first++));
do
    for((second=$first+4;second<=15;second++));
    do
        echo "Cut combination: [$first, $second]"
        /notebooks/speedup-gen-surgery/scripts/run_experiments.sh hyperparams SuperResolution $first $second 04974.png
    done
done