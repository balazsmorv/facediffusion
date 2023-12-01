import torch
from pathlib import Path
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from schedulers import linear_beta_schedule, cosine_beta_schedule
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

    model_path = '/home/oem/facediffusion/results/model_epoch_399ema.pth'
    batch_size = 96
    dataset_path = "/home/oem/FDF/val"
    beta_schedule = ""


inference_params = DDIM_Inference_Params()

@torch.inference_mode()
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

@torch.inference_mode()
def generalized_steps(x, seq, model, b, init_images: torch.Tensor):
    n = x.size(0)
    eta = 0.0
    seq_next = [-1] + list(seq[:-1])
    xt = x
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        et = model(xt, t)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # denoised observation, which is a predicition of x_0, given x_t
        c1 = (
            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() # sigma_t
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt() # part of the "direction pointing to x_t"
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et # get x_t-1 from x_t
        xt_next = torch.where(init_images == 0, xt_next, init_images)
        xt = xt_next

    return xt


@torch.inference_mode()
def sample_image(timesteps: int, device: str, model_fn, init_images: torch.Tensor):
    skip = inference_params.timesteps // timesteps
    seq = range(0, inference_params.timesteps, skip) # tau-s
    random_noise = torch.randn(size=(init_images.shape[0], inference_params.channels, inference_params.image_size, inference_params.image_size), device=device)
    input_images = torch.where(init_images == 0, random_noise, init_images)
    betas = linear_beta_schedule(inference_params.timesteps).to(device)
    images = generalized_steps(x=input_images, seq=seq, model=model_fn, b=betas, init_images=init_images)
    return images


if __name__ == '__main__':

    img_transform = Compose([
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Resize(inference_params.image_size),
        CenterCrop(inference_params.image_size),
        Lambda(lambda t: (t * 2) - 1),
    ])

    mask_transform = Compose([
        Resize(inference_params.image_size),
        CenterCrop(inference_params.image_size),
    ])

    dataset = FDF256Dataset(dirpath=inference_params.dataset_path, load_keypoints=True,
                            img_transform=img_transform, mask_transform=mask_transform, load_masks=True)
    dataloader = DataLoader(dataset, batch_size=inference_params.batch_size, shuffle=False, num_workers=8,
                            prefetch_factor=1, persistent_workers=False, pin_memory=False)

    os.makedirs('generated_images_ddim', exist_ok=True)
    os.makedirs('original_images', exist_ok=True)
    os.makedirs('masked_images', exist_ok=True)

    if torch.cuda.is_available(): device = 'cuda'; print('CUDA is available')
    elif torch.backends.mps.is_available(): device = 'mps'; print('MPS is available')
    else: device = 'cpu'; print('CPU is available')

    model = Unet(
        dim=inference_params.image_size,
        channels=inference_params.channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2 if dataset.load_keypoints else None)
    )

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(Path(inference_params.model_path), map_location=torch.device(device)))
    model.eval()

    for index, og_data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Image generation on the validation set'):
        og_keypoints = og_data['keypoints'].to(device)
        og_keypoints = rearrange(og_keypoints.view(og_data['img'].shape[0], -1), "b c -> b c 1 1")
        model_fn = partial(model, x_self_cond=og_keypoints)
        img2inpaint = og_data['img'].to(device) * og_data['mask'].to(device)

        samples = sample_image(timesteps = 16, device = device, model_fn = model_fn, init_images=img2inpaint)
        samples = rearrange(samples, "b c h w -> b h w c")
        for i, sample in enumerate(samples):
            number = index * inference_params.batch_size + i
            plt.imsave(f'ddim_generated_images_16/{number}.jpeg', np.asarray((sample.to('cpu') + 1) / 2 * 255, dtype=np.uint8))
