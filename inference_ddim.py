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
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from skimage.util import montage
import shutil
import os

class DDIM_Inference_Params:
    timesteps = 1024
    channels = 3
    image_size = 64

    model_path = '/Users/balazsmorvay/Downloads/Azure VM/facediffusion/model_weights/model_epoch_399ema.pth'
    batch_size = 32

inference_params = DDIM_Inference_Params()

class Schedule:
    def __init__(self, steps):
        self.steps = steps

        # define beta schedule
        self.betas: torch.Tensor = linear_beta_schedule(timesteps=steps)

        # define alphas
        self.alphas: torch.Tensor = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - self.alphas_cumprod_prev)

        # calculations for posterior q(x_{t-1} | x_t, x_0) variance (beta)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

@torch.inference_mode()
def predict_xstart_from_eps(schedule, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
            extract(schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

@torch.inference_mode()
def ddim_sample_mean(schedule, model, x, t):
    """
    Returns the mean of the ddim reverse process q(x_t | x_t-1, x_0)
    """
    sqrt_alphas_cumprod_prev_extr = extract(schedule.sqrt_alphas_cumprod_prev, t, x.shape)
    sqrt_one_minus_alphas_cumprod_prev_extr = extract(schedule.sqrt_one_minus_alphas_cumprod_prev, t, x.shape)

    pred_noise = model(x, t)
    pred_x0 = predict_xstart_from_eps(schedule, x, t, pred_noise)

    pred_mean = pred_x0 * sqrt_alphas_cumprod_prev_extr + sqrt_one_minus_alphas_cumprod_prev_extr * pred_noise
    return pred_mean

@torch.inference_mode()
def p_sample_loop(model, shape, device, img2inpaint=None):
    b = shape[0]
    schedule = Schedule(inference_params.timesteps)
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape if img2inpaint is None else img2inpaint.shape, device=device)
    if img2inpaint is not None:
        img = torch.where(img2inpaint == 0, img, img2inpaint)
    imgs = []

    for i in tqdm(reversed(range(0, inference_params.timesteps)), desc='reverse process time step', total=inference_params.timesteps):
        img = ddim_sample_mean(schedule=schedule,
                               model=model,
                               x=img,
                               t=torch.full((b,), i, device=device, dtype=torch.long))

        if img2inpaint is not None:
            img = torch.where(img2inpaint == 0, img, img2inpaint)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.inference_mode()
def sample(model, image_size, batch_size, channels, device, img2inpaint=None):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device=device,
                         img2inpaint=img2inpaint)



if __name__ == '__main__':

    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('original_images', exist_ok=True)
    os.makedirs('masked_images', exist_ok=True)

    if torch.cuda.is_available(): device = 'cuda'; print('CUDA is available')
    elif torch.backends.mps.is_available(): device = 'mps'; print('MPS is available')
    else: device = 'cpu'; print('CPU is available')

    model = Unet(
        dim=inference_params.image_size,
        channels=inference_params.channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2) #if dataset.load_keypoints else None)
    )

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(Path(inference_params.model_path), map_location=torch.device(device)))
    model.eval()

    #og_keypoints = torch.tensor(data=[[]])
    #model_fn = partial(model, x_self_cond=og_keypoints)

    samples = sample(model,
                     image_size=inference_params.image_size,
                     batch_size=inference_params.batch_size,
                     channels=inference_params.channels,
                     device=device,
                     img2inpaint=None)

    for i in range(inference_params.batch_size):
        image = rearrange(samples[-1][i], 'c h w -> h w c')
        plt.imsave(f'generated_images/{i}.jpeg', np.asarray((image + 1) / 2 * 255, dtype=np.uint8))

