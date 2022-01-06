"""Experiment Settings"""
n_measure_64 = [100, 200, 300, 400, 600, 1000, 2000, 3000, 4000, 6000]
n_measure_128 = [400, 800, 1200, 1600, 2400, 4000, 8000, 12000, 16000, 24000]

forward_models = {
    'began_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
    },
    'biggan_inv': {
        'InpaintingSquare': [{
            'mask_size': 256
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
    },
    'mgan_biggan_inv': {
        'InpaintingSquare': [{
            'mask_size': 256
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
    },
    'dcgan_inv': {
        'InpaintingSquare': [{
            'mask_size': 14
        }],
    },
    'vanilla_vae_inv': {
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'iagan_dcgan_inv': {
        'InpaintingSquare': [{
            'mask_size': 14
        }],
    },
    'iagan_began_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
    },
    'iagan_biggan_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
    },
    'mgan_dcgan_inv': {
        'InpaintingSquare': [{
            'mask_size': 14
        }],
    },
    'mgan_began_inv': {
        'InpaintingSquare': [{
            'mask_size': 32
        }],
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
    },
}

baseline_settings = {
    'lasso-dct-64': {
        'img_size': 64,
        'lasso_coeff': [0.01] * len(n_measure_64),
        'n_measure': n_measure_64,
    },
    'lasso-dct-128': {
        'img_size': 128,
        'lasso_coeff': [0.01] * len(n_measure_128),
        'n_measure': n_measure_128,
    },
}

recovery_settings = {
    'began_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 80,
        'z_lr': 0.8,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [1, 2],
        'limit': [1],
    },
    'biggan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1.5,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'restarts': 3,
        'n_cuts_list': [0, 7],
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
    'vanilla_vae_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'iagan_dcgan_inv': {
        'optimizer': 'adam',
        'z_init_mode': ['clamped_normal'],
        'z_steps1': 1600,
        'z_steps2': 1200,
        'z_lr1': 0.1,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 3,
        'n_cuts_list': [0,1],
        'limit': [1],
    },
    'iagan_began_inv': {
        'optimizer': 'adam',
        'z_init_mode': ['clamped_normal'],
        'z_steps1': 1600,
        'z_steps2': 600,
        'z_lr1': 1,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 3,
        'n_cuts_list': [0,2],
        'limit': [1],
    },
    'iagan_biggan_inv': {
        'optimizer': 'adam',
        'z_init_mode': ['clamped_normal'],
        'z_steps1': 20,
        'z_steps2': 10,
        'z_lr1': 1.5,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 2,
        'n_cuts_list': [0,7],
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
    'mgan_began_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 80,
        'z_lr': 0.8,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 20,
        'restarts': 3,
#         'cuts': [[1,13], [1,14], [1,15], [2,6],[2,9],[2,11],[2,15], [3,9],[3,15],[4,9],[4,15]]
    },
     'mgan_biggan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1.5,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 20,
        'restarts': 3,
#         'cuts': [[1,6],[1,14],[3,6],[3,14],[5,6],[5,14],[7,10],[7,14],[8,15],[9,15]]
    },
    
}
