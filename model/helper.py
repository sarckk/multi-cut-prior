import torch.nn as nn

class ConvBlock(nn.Module):
    """
    All convs are created with:
    conv(in_channel, out_channel, kernel, stride, pad, bias)
    """
    def __init__(self, in_ch, out_ch, k, s, p, b=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      k,
                      s,
                      p,
                      bias=b), nn.ELU())

    def forward(self, x):
        return self.net(x)



class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
