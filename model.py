import torch
from torch import nn
from functools import partial
from resnet_block import ResnetBlock
from network_helper import default, Upsample, Downsample, Residual
from embeddings import SinusoidalPositionEmbeddings
from normalization import PreNorm
from attention import Attention, LinearAttention

class Unet(nn.Module):
    """
    the network takes a batch of noisy images of shape (batch_size, num_channels, height, width) and a batch of noise levels of shape (batch_size, 1) as input, and returns a tensor of shape (batch_size, num_channels, height, width)
The network is built up as follows:
first, a convolutional layer is applied on the batch of noisy images, and position embeddings are computed for the noise levels
next, a sequence of downsampling stages are applied. Each downsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + a downsample operation
at the middle of the network, again ResNet blocks are applied, interleaved with attention
next, a sequence of upsampling stages are applied. Each upsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + an upsample operation
finally, a ResNet block followed by a convolutional layer is applied.
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition_dim=None,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups, 
                              condition_dim=self_condition_dim if self_condition_dim > 0 else None)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.self_condition = (self_condition_dim is not None) and (self_condition_dim > 0)
        if self.self_condition:
            self.cond_embed = nn.Sequential(
                nn.Conv2d(self_condition_dim, init_dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, kernel_size=(1, 1))
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        x = self.init_conv(x) # Conv layer is applied on the batch of noisy images
        r = x.clone()

        t = self.time_mlp(time) # position embeddings are computed for the noise levels

        h = []


        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, x_self_cond) # resnet block
            h.append(x)

            x = block2(x, t, x_self_cond) # resnet block
            x = attn(x) # attention
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, x_self_cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, x_self_cond)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, x_self_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, x_self_cond)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, x_self_cond)
        return self.final_conv(x)
