forward_models = {
    'Denoising': {
      'sigma': 0.2  
    },
    'InpaintingIrregular': {
        'mask_name': '04244.png'
    }, 
    'InpaintingScatter': {
        'fraction_kept': 0.1
    },
    'SuperResolution': {
        'scale_factor': 0.25,
        'mode': 'bilinear',
        'align_corners': True
    },
}
