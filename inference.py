import torch
from pathlib import Path
import matplotlib.pyplot as plt
from schedulers import linear_beta_schedule
from model import Unet
import torch.nn.functional as F
from tqdm.auto import tqdm
from train import extract, sample
from PIL import Image

experiment_name = "MNIST_model"
channels = 3
torch.manual_seed(0)
timesteps = 300
image_size = 64

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


if __name__ == '__main__':

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model = torch.nn.DataParallel(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(Path("./results/" + experiment_name + ".pth")))
    model.eval()
    # inference
    samples = sample(model, image_size=image_size, batch_size=1, channels=channels)
    # show a random one
    random_index = 0
    image = samples[-1][random_index].reshape(image_size, image_size, channels)[:,:,0]
    print(image)
    plt.imsave('example.jpeg', image, cmap='gray')
    plt.imshow(image)
    plt.show()
