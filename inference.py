import torch
from pathlib import Path
import matplotlib.pyplot as plt
from schedulers import linear_beta_schedule
from model import Unet
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from network_helper import extract
from fdh256_dataset import FDF256Dataset
from einops import rearrange
from torch.utils.data import DataLoader
from functools import partial
import numpy as np

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, shape, device="cuda"):

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, device="cuda"):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device="cuda")


experiment_name = "model_548"
channels = 3
torch.manual_seed(0)
timesteps = 1000
image_size = 256

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

dataset = FDF256Dataset(dirpath="/home/oem/FDF/train", load_keypoints=True, transform=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                        prefetch_factor=1, persistent_workers=False, pin_memory=False)


device="cuda"

if __name__ == '__main__':

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2 if dataset.load_keypoints else None)
    )
    model = torch.nn.DataParallel(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(Path("./results/" + experiment_name + ".pth")))
    model.eval()

    og_data = next(iter(dataloader))
    og_keypoints = og_data['keypoints'].to(device)
    og_keypoints = rearrange(og_keypoints.view(og_data['img'].shape[0], -1), "b c -> b c 1 1")
    model_fn = partial(model, x_self_cond=og_keypoints)

    batch_size = 10

    # inference
    samples = sample(model_fn, image_size=image_size, batch_size=batch_size, channels=channels) # list of 1000 ndarrays of shape (batchsize, 3, 64, 64)

    plt.imsave('og_image.jpeg', og_data['img'][0].numpy())

    for i in range(batch_size):
        image = rearrange(samples[-1][i], 'c h w -> h w c')
        plt.imsave(f'example_{i}.jpeg', np.asarray((image + 1) / 2 * 255, dtype=np.uint8))
