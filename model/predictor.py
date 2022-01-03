import torch
import torch.nn as nn

from .helper import ConvBlock, conv


class PredictorDCGAN(nn.Module):
    def __init__(self, nc = 3, ndf = 64):
        super(PredictorDCGAN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            conv(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            conv(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            conv(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

    def forward(self, x):
        output = self.main(x)
        return output
    
    

class PredictorBEGAN(nn.Module):
    def __init__(self):
        super(PredictorBEGAN, self).__init__()
        self.num_channel = 32 # this ensures that the final number of channels is 32 * 4 = 128
        self.layers = nn.Sequential(
            ConvBlock(3, self.num_channel, 3, 1, 1),
            ConvBlock(self.num_channel, self.num_channel, 3, 1, 1),
            ConvBlock(self.num_channel, self.num_channel, 3, 1, 1),
            nn.Conv2d(self.num_channel, self.num_channel, 1, 1, 0),
            nn.AvgPool2d(2, 2),
            
            ConvBlock(self.num_channel, self.num_channel, 3, 1, 1),
            ConvBlock(self.num_channel, self.num_channel, 3, 1, 1),
            nn.Conv2d(self.num_channel, 2*self.num_channel, 1, 1, 0),
            nn.AvgPool2d(2, 2),
            
            ConvBlock(2*self.num_channel, 2*self.num_channel, 3, 1, 1),
            ConvBlock(2*self.num_channel, 2*self.num_channel, 3, 1, 1),
            nn.Conv2d(2*self.num_channel, 3*self.num_channel, 1, 1, 0),
            nn.AvgPool2d(2, 2),
            
            ConvBlock(3*self.num_channel, 3*self.num_channel, 3, 1, 1),
            ConvBlock(3*self.num_channel, 3*self.num_channel, 3, 1, 1),
            nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.layers(x)
