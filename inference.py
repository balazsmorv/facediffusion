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


INFERENCE_CFG = {
    # Model and train parameters
    'timesteps': 1024,
    'channels': 3,

    # Dataset params
    'dataset_pth': "/home/oem/FDF/val",
    'load_keypoints': True,
    'image_size': 64,
    'batch_size': 2,

    # Logging parameters
    'experiment_name': 'model_epoch_399ema',
    'save_montage': False
}

class Schedule:
    def __init__(self, steps):
        self.steps = steps

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=steps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - self.alphas_cumprod_prev)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

def predict_xstart_from_eps(schedule, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
            extract(schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

@torch.no_grad()
def ddim_sample(schedule, model, x, t):
    sqrt_alphas_cumprod_prev_extr = extract(schedule.sqrt_alphas_cumprod_prev, t, x.shape)
    sqrt_one_minus_alphas_cumprod_prev_extr = extract(schedule.sqrt_one_minus_alphas_cumprod_prev, t, x.shape)

    pred_noise = model(x, t)
    pred_x0 = predict_xstart_from_eps(schedule, x, t, pred_noise)

    pred_mean = pred_x0 * sqrt_alphas_cumprod_prev_extr + sqrt_one_minus_alphas_cumprod_prev_extr * pred_noise
    return pred_mean

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
def p_sample_loop(model, shape, device="cuda", img2inpaint=None):
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape if img2inpaint is None else img2inpaint.shape, device=device)
    img = torch.where(img2inpaint == 0, img, img2inpaint)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        #img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        img = ddim_sample(schedule=Schedule(INFERENCE_CFG['timesteps']), model=model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long))
        if img2inpaint is not None:
            img = torch.where(img2inpaint == 0, img, img2inpaint)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, device="cuda", img2inpaint=None):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device="cuda",
                         img2inpaint=img2inpaint)


os.makedirs('generated_images', exist_ok=True)
os.makedirs('original_images', exist_ok=True)
os.makedirs('masked_images', exist_ok=True)

experiment_name = INFERENCE_CFG['experiment_name']
channels = INFERENCE_CFG['channels']
torch.manual_seed(0)
timesteps = INFERENCE_CFG['timesteps']
image_size = INFERENCE_CFG['image_size']

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

img_transform = Compose([
    ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
    Resize(image_size),
    CenterCrop(image_size),
    Lambda(lambda t: (t * 2) - 1),
])

mask_transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
])

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    # ToPILImage(),
])

dataset = FDF256Dataset(dirpath=INFERENCE_CFG['dataset_pth'], load_keypoints=INFERENCE_CFG['load_keypoints'],
                        img_transform=img_transform, mask_transform=mask_transform, load_masks=True)
dataloader = DataLoader(dataset, batch_size=INFERENCE_CFG['batch_size'], shuffle=False, num_workers=1,
                        prefetch_factor=1, persistent_workers=False, pin_memory=False)

device = "cuda"

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

    batch_size = INFERENCE_CFG['batch_size']

    if dataset.load_masks:
        img2inpaint = og_data['img'].to(device) * og_data['mask'].to(device)
    else:
        img2inpaint = None

    # inference
    samples = sample(model_fn, image_size=image_size, batch_size=batch_size, channels=channels,
                     img2inpaint=img2inpaint)  # list of 1000 ndarrays of shape (batchsize, 3, 64, 64)

    reversed_imgs = []
    for img_idx in range(og_data['img'].shape[0]):
        reversed_imgs.append(reverse_transform(og_data['img'][img_idx]))
    reversed_imgs = np.stack(reversed_imgs, 0)
    reversed_masked_imgs = reversed_imgs * (
        og_data['mask'].permute(0, 2, 3, 1).cpu().numpy() if dataset.load_masks else 1)
    og_masked_batch = montage(reversed_masked_imgs, channel_axis=3)
    # plt.imsave('og_image.jpeg', reverse_transform(og_data['img'][0]) * (og_data['mask'][0].permute(1, 2, 0).cpu().numpy() if dataset.load_masks else 1))
    plt.imsave('og_masked_batch.png', og_masked_batch)

    og_batch = montage(reversed_imgs, channel_axis=3)
    plt.imsave('og_batch.png', og_batch)

    if INFERENCE_CFG['save_montage']:
        gen_batch = montage(np.asarray(((samples[-1].transpose(0, 2, 3, 1) + 1) / 2) * 255, dtype=np.uint8),
                            channel_axis=3)
        plt.imsave('gen_batch.png', gen_batch)
    else:
        for i in range(batch_size):
            image = rearrange(samples[-1][i], 'c h w -> h w c')
            plt.imsave(f'generated_images/{i}.jpeg', np.asarray((image + 1) / 2 * 255, dtype=np.uint8))

