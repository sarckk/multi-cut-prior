import torch
import torch.nn as nn

from pytorch_pretrained_biggan import BigGAN
from pytorch_pretrained_biggan.model import GenBlock, Generator


# Modified from
# https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/pytorch_pretrained_biggan/model.py#L289
def patch_biggan_forward(self, z1, z2, truncation=0.4, n_cuts=0, end=None):
    return self.generator(z1, z2, truncation=truncation, n_cuts=n_cuts, end=end)


# Modified from
# https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/pytorch_pretrained_biggan/model.py#L228
def patch_generator_forward(self, z1, z2=None, truncation=0.4, n_cuts=0, end=None):
    """
    If n_cuts is 0, z1 is the 128 latent vector ("z"), and z2 is the
    128 class label embedding ("embed")

    If n_cuts >= 1, z1 is some high-dim latent vector ("z"), and z2 is
    the concat ("cond_vector")
    
    end is from 0 to 17 -> number of layers (no. of input_shapes is 1 more than no of layers)
    """
    assert end >= 0 and end <= 17
    
    completed = 0
    
    if end == completed:
        return z1
    
    if n_cuts == 0:
        # assert z1.shape[1] == z2.shape[1] == 128

        z2 = torch.cat((z1, z2), dim=1)
        z1 = self.gen_z(z2)

        z1 = z1.view(-1, 4, 4, 16 * 128)
        z1 = z1.permute(0, 3, 1, 2).contiguous()
        # print("first", z1.shape)
    else:
        n_cuts -= 1
    
    completed += 1 
    
    #len(self.layers) is 15
    # min.value of end is 1 

    for i, layer in enumerate(self.layers):
        if end == completed:
            break
            
        if n_cuts == 0:
            if isinstance(layer, GenBlock):
                z1 = layer(z1, z2, truncation)
                # print("-->", z1.shape)
            else:
                z1 = layer(z1)
                # print("-->", z1.shape)
        else:
            n_cuts -= 1
        
        completed +=1
    
    if end == completed:
        return z1
        
    if n_cuts == 0:
        z1 = self.bn(z1, truncation)
        z1 = self.relu(z1)
        z1 = self.conv_to_rgb(z1)
        z1 = z1[:, :3, ...]
        z1 = self.tanh(z1)
    else:
        n_cuts -= 1
    
    # print("-->", z1.shape)
    completed +=1

    assert n_cuts == 0
    # assert completed == end
    return z1


class BigGanSkip(nn.Module):
    rescale = False

    def __init__(self):
        super().__init__()
        self.biggan = BigGAN.from_pretrained('biggan-deep-128')
        self.image_size = 512
    

        # Monkey patch forward methods for n_cuts
        BigGAN.forward = patch_biggan_forward
        Generator.forward = patch_generator_forward
        # NOTE - because each resblock reduces channels and
        # then increases, we cannot skip into the middle.
        # If we did, we would have no way to add channels
        # to the skip connection ("x0")
        

        # 18 available for 512
#         self.input_shapes = [
#             ((128, ), (128, )),  # Raw input shape
#             ((2048, 4, 4), (256, )),  # Linear
#             ((2048, 4, 4), (256, )),  # Block
#             ((2048, 8, 8), (256, )),  # Block Up
#             ((2048, 8, 8), (256, )),  # Block
#             ((1024, 16, 16), (256, )),  # Block Up
#             ((1024, 16, 16), (256, )),  # Block
#             ((1024, 32, 32), (256, )),  # Block Up
#             ((1024, 32, 32), (256, )),  # Block
#             ((512, 64, 64), (256, )),  # Block Up
#             ((512, 64, 64), (256, )),  # Self-Attention block
#             ((512, 64, 64), (256, )),  # Block
#             ((256, 128, 128), (256, )),  # Block Up
#             ((256, 128, 128), (256, )),  # Block
#             ((128, 256, 256), (256, )),  # Block Up
#             ((128, 256, 256), (256, )),  # Block
#             ((128, 512, 512), (256, )),  # Block Up
#             ((3, 512, 512), ()),  # Final Conv
#         ]
        
        # end can be 0 to 13
        self.input_shapes = [
            ((128, ), (128, )),  # Raw input shape
            ((2048, 4, 4), (256, )),  # Linear
            ((2048, 4, 4), (256, )),  # Block
            ((2048, 8, 8), (256, )),  # Block Up
            ((2048, 8, 8), (256, )),  # Block
            ((1024, 16, 16), (256, )),  # Block Up
            ((1024, 16, 16), (256, )),  # Block
            ((512, 32, 32), (256, )),  # Block Up
            ((512, 32, 32), (256, )),  # Block
            ((256, 64, 64), (256, )),  # Block Up
            ((256, 64, 64), (256, )),  # Self-Attention block
            ((256, 64, 64), (256, )),  # Block
            ((128, 128, 128), (256, )),  # Block Up
            ((3, 128, 128), ()),  # Final Conv
        ]
        # self._check_input_shapes()
    

    def _check_input_shapes(self):
        for n_cuts, (z1, z2) in enumerate(self.input_shapes):
            z1 = torch.randn(1, *z1)
            z2 = torch.randn(1, *z2)
            res = self.forward(z1, z2, n_cuts=n_cuts)
            print(n_cuts, z1.shape, z2.shape, res.shape[1:])
        return

    def forward(self, z1, z2, truncation=0.4, n_cuts=0, end=None):
        """
        z1 - represents the result of skipping layers
        z2 - represents the concat of their latent vector (128) +
             their class embedding (128)
        """
        if end is None:
            end = len(self.input_shapes) - 1
        return self.biggan(z1, z2, truncation=truncation, n_cuts=n_cuts,end=end)

    def __str__(self):
        return 'BigGanSkip'

    

if __name__ == '__main__':
    # Test the start and end 
    model = BigGanSkip()
    model.cuda()
    
    for end in range(0, 18):
        # end can be 0 to 17 -> 0 means just return z_1
        for start in range(0, end+1):
            # start can be 0 to 17 -> 17 means skip all the layers 
            z1_shape, z2_shape = model.input_shapes[start]
            z1 = torch.randn(1, *z1_shape)
            z1 = z1.cuda()
            z2 = torch.randn(1, *z2_shape)
            z2 = z2.cuda()
            res = model.forward(z1, z2, n_cuts=start, end=end)
            print('actual: ', res.shape)
            print('expected:', model.input_shapes[end][0])
            assert res.squeeze(0).shape == model.input_shapes[end][0], f"first part failed at {start},{end}"
            print('====================================')
            z2 = torch.cat((z1,z2),dim=1) if start == 0 and end > 0 else z2
            z3 = torch.randn(1, *model.input_shapes[end][1]).cuda()
            final = model.forward(res, z3, n_cuts=end)
            assert final.squeeze(0).shape == model.input_shapes[-1][0], f"second part failed at {start},{end}"
    
    print("verified everything!")
        