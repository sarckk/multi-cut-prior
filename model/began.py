import torch
import torch.nn as nn

from .helper import View, ConvBlock

    
class BEGAN_Decoder(nn.Module):
    rescale=False
    def __init__(self):
        super(BEGAN_Decoder, self).__init__()
        self.ch = 128
        self.latent_dim = 64
        self.scale_size = 128
        self.initial_size = 8
        
        self.layers = nn.ModuleList([
             nn.Sequential(
                nn.Linear(self.latent_dim,
                          self.initial_size**2 * self.ch,
                          bias=True),
                View((-1, self.ch, self.initial_size, self.initial_size))
             ),
            
            # first block
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            
            # second block
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            
            # third block
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            
            # fourth block
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            
            # fifth block
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            ConvBlock(self.ch, self.ch, 3, 1, 1, True),
            
            # last block
            nn.Sequential(
                nn.Conv2d(self.ch, 3, 3, 1, 1),
                nn.Tanh()
            )
        ])
        
        self.input_shapes = [
            # Raw input shape
            ((self.latent_dim, ), ()),

            # Skip Linear+View()
            ((128, 8, 8), ()),

            # First block
            ((128, 8, 8), ()),
            ((128, 8, 8), ()),

            # Second conv
            ((128, 16, 16), ()),
            ((128, 16, 16), ()),
            ((128, 16, 16), ()),

            # Third conv
            ((128, 32, 32), ()),
            ((128, 32, 32), ()),
            ((128, 32, 32), ()),
             
            # Fourth conv
            ((128, 64, 64), ()),
            ((128, 64, 64), ()),
            ((128, 64, 64), ()),
             
            # Fifth conv
            ((128, 128, 128), ()),
            ((128, 128, 128), ()),
            ((128, 128, 128), ()),

            # Skip the whole net
            ((3, 128, 128), ()),
        ]
        
        # self._check_input_shapes()
        
    def _check_input_shapes(self):
        for n_cuts, (x1_shape, x2_shape) in enumerate(self.input_shapes):
            print(n_cuts)
            x1 = torch.randn(1, *x1_shape)
            if n_cuts <= 1:
                x2 = None
            else:
                x2 = torch.randn(1, *x2_shape)
            res = self.forward(x1, x2, n_cuts)
            print(x1.shape, () if n_cuts <= 1 else x2.shape, res.shape[1:])
            

    def forward(self, z, z2=None, n_cuts=0, end=-1):
        if end == -1:
            end = len(self.layers)
        for i, layer in enumerate(self.layers[n_cuts:end]):
            z = layer(z)
        return z
    
    def __str__(self):
        return f'Began.Gen128.latent_dim={self.latent_dim}'