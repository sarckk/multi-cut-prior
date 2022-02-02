#!/usr/bin/bash

for((i=1;i<=15;i++));
do
    echo "==> mGANPrior: cut at $i"
    /notebooks/speedup-gen-surgery/scripts/run_experiments.sh hyperparams SuperResolution 0 $i 04974.png
done
