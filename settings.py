"""Experiment Settings"""

# other inpainting candidates: 4129, 04974.png
#     'InpaintingIrregular': [{
#         'mask_name': '03874.png'
#     }, {
#         'mask_name': '03600.png'
#     }, {
#         'mask_name': '03437.png'
#     }, {
#         'mask_name': '04138.png'
#     }, {
#         'mask_name': '04244.png'

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
