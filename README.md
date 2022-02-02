# Generator Surgery with Multi-Code GAN Prior for Image Restoration

## Sample commands
```python
# default BEGAN
./scripts/run_experiments.sh project-name InpaintingIrregular 0 -1

# BEGAN with surgery
./scripts/run_experiments.sh project-name InpaintingIrregular 2 -1 04974.png

# BEGAN using mGANPrior
./scripts/run_experiments.sh project-name InpaintingScatter 0 14 04974.png

# Our method (GS + mGANPrior)
./scripts/run_experiments.sh project-name InpaintingScatter 1 14 04974.png
```

## Acknowledgements
The code in this repo was forked from https://github.com/nik-sm/generator-surgery (Generator Surgery for Compressed Sensing
by Jung Yeon Park\*, Niklas Smedemark-Margulies\*, Max Daniels, Rose Yu, Jan-Willem van de Meent, and Paul Hand) and later modified.

Pretrained weights for pretrained modles come from:

DCGAN - https://github.com/Natsu6767/DCGAN-PyTorch

BEGAN - https://github.com/zhusiling/BEGAN/tree/master/trained_models/128_tanh/models

BigGAN - https://github.com/huggingface/pytorch-pretrained-BigGAN

