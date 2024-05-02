import torch
from torch import nn
import math
import matplotlib.pyplot as plt

class SinusoidalPositionEmbeddings(nn.Module):
    """
    takes a tensor of shape (batch_size, 1) as input (i.e. the noise levels of several noisy images in a batch),
    and turns this into a tensor of shape (batch_size, dim), with dim being the dimensionality of the position embeddings.
    This is then added to each residual block, as we will see further.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

if __name__ == '__main__':
    embedding = SinusoidalPositionEmbeddings(5)
    out = embedding(torch.tensor(data=([1.0])))
    print(out)
    print(out.shape)
    plt.plot(out[0])
    plt.show()