#!/usr/bin/bash

for((i=1;i<=6;i++));
do
    echo "Cut at $i"
    /notebooks/speedup-gen-surgery/scripts/run_experiments.sh hyperparams InpaintingIrregular $i -1 04974.png
done
