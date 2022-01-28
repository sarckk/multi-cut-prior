"""Experiment Settings"""

# other inpainting candidates: 4129
began_experiments = {
    'InpaintingIrregular': [{
        'mask_name': '04974.png'
    }]
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
#     }],
#     'InpaintingScatter': [{
#         'fraction_kept': 0.1
#     }],
#     'SuperResolution': [{
#         'scale_factor': 0.25,
#         'mode': 'bilinear',
#         'align_corners': True
#     }],
}

began_shared_settings = {
    'optimizer': 'lbfgs',
    'n_steps': 40,
    'z_lr': 1,
    'z_init_mode': ['clamped_normal'],
    'restarts': 1,
    'limit': [1],
}

forward_models = {
    'biggan_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.05
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'mgan_biggan_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.05
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'dcgan_inv': {
        'InpaintingSquare': [{
            'mask_size': 14
        }],
    },
    'mgan_dcgan_inv': {
        'InpaintingSquare': [{
            'mask_size': 14
        }],
    },
    'began_inv': began_experiments,
    'mgan_began_inv': began_experiments,
}

recovery_settings = {
    'biggan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'restarts': 3,
        'n_cuts_list': [6,7],
    },
    'dcgan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 20,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [1],
        'limit': [1],
    },
    'mgan_dcgan_inv': {
        'optimizer': 'adam',
        'n_steps': 5000,
        'z_lr': 3e-2,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 10,
        'restarts': 3,
    },
    'began': {
        **began_shared_settings,
    },
     'mgan_biggan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 40,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 20,
        'restarts': 1,
#         'cuts': [[1,6],[1,14],[3,6],[3,14],[5,6],[5,14],[7,10],[7,14],[8,15],[9,15]]
    },
    
}
