from zipfile import BadZipFile
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
import os, psutil
import subprocess as sp
from time import time_ns, time
import sys
import pandas as pd

PLATFORM = sys.argv[3]

if PLATFORM == 'jetson':
    from jtop import jtop
    jetson = jtop()
    jetson.start()

INFERENCE_CFG = {
    # Model and train parameters
    'timesteps': 1024,
    'channels': 3,

    # Dataset params
    'dataset_pth': "./dataset/FDF/train",
    'load_keypoints': True,
    'image_size': 64,
    'batch_size': int(sys.argv[2]) if len(sys.argv) > 1 else 10,

    # Logging parameters
    'experiment_path': sys.argv[1] if len(sys.argv) > 1 else 'results/model_epoch_399.pth',
    'save_montage': False
}

logs = np.zeros((INFERENCE_CFG['timesteps'], 4))

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)][0]
    return memory_use_values

def get_memory_usage():
    if PLATFORM == 'jetson':
        for process in jetson.processes:
            if len(process) == 0:
                continue
            if process[-1] == 'python3':
                cpu_mem = round(process[-3] / 1000)
                gpu_mem = round(process[-2] / 1000)
                return cpu_mem, gpu_mem
    elif PLATFORM == 'rpi':
        cpu_mem = round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        return cpu_mem, 0
    else:
        cpu_mem = round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        gpu_mem = get_gpu_memory()
        return cpu_mem, gpu_mem
    return 0,0

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
        step_start = time_ns()
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        step_time = round((time_ns() - step_start) / 1000**2)
        step_freq = round(1 / step_time * 1000)
        if sys.argv[4] == 'mem':
            cpu_mem, gpu_mem = get_memory_usage()
        else:
            cpu_mem, gpu_mem = 0,0
        # print(step_time, step_freq, cpu_mem, gpu_mem)
        logs[i] = np.array((step_time, step_freq, cpu_mem, gpu_mem))
        if img2inpaint is not None:
            img = torch.where(img2inpaint == 0, img, img2inpaint)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, device="cuda", img2inpaint=None):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device=device,
                         img2inpaint=img2inpaint)


#os.makedirs('generated_images')
#os.makedirs('original_images')
#os.makedirs('masked_images')

experiment_path = INFERENCE_CFG['experiment_path']
channels = INFERENCE_CFG['channels']
torch.manual_seed(0)
timesteps = INFERENCE_CFG['timesteps']
image_size = INFERENCE_CFG['image_size']
device= "cuda" if torch.cuda.is_available() else "cpu"

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
dataloader = DataLoader(dataset, batch_size=INFERENCE_CFG['batch_size'], shuffle=False, num_workers=20,
                        prefetch_factor=1, persistent_workers=True, pin_memory=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


def perform_demo_inference():
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2 if dataset.load_keypoints else None)
    )
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    model.load_state_dict(torch.load(experiment_path))
    model.eval()

    og_data = next(iter(dataloader))
    og_keypoints = og_data['keypoints'].to(device)
    og_keypoints = rearrange(og_keypoints.view(og_data['img'].shape[0], -1), "b c -> b c 1 1")
    model_fn = partial(model, x_self_cond=og_keypoints)

    batch_size = INFERENCE_CFG['batch_size']

    print(f"Batch size: {batch_size}")

    if dataset.load_masks:
        img2inpaint = og_data['img'].to(device) * og_data['mask'].to(device)
    else:
        img2inpaint = None
    samples = sample(model_fn, image_size=image_size, batch_size=batch_size, channels=channels,
                    img2inpaint=img2inpaint, device=device)  # list of 1000 ndarrays of shape (batchsize, 3, 64, 64)
    
    reversed_imgs = []
    for img_idx in range(og_data['img'].shape[0]):
        reversed_imgs.append(reverse_transform(og_data['img'][img_idx]))
    reversed_imgs = np.stack(reversed_imgs, 0)
    reversed_masked_imgs = reversed_imgs * (
        og_data['mask'].permute(0, 2, 3, 1).cpu().numpy() if dataset.load_masks else 1)
    og_masked_batch = montage(reversed_masked_imgs, channel_axis=3)
    # plt.imsave('og_image.jpeg', reverse_transform(og_data['img'][0]) * (og_data['mask'][0].permute(1, 2, 0).cpu().numpy() if dataset.load_masks else 1))
    plt.imsave('results/og_masked_batch.png', og_masked_batch)

    og_batch = montage(reversed_imgs, channel_axis=3)
    plt.imsave('results/og_batch.png', og_batch)

    if INFERENCE_CFG['save_montage']:
        gen_batch = montage(np.asarray(((samples[-1].transpose(0, 2, 3, 1) + 1) / 2) * 255, dtype=np.uint8),
                            channel_axis=3)
        plt.imsave('results/gen_batch.png', gen_batch)
    else:
        for i in range(batch_size):
            image = rearrange(samples[-1][i], 'c h w -> h w c')
            plt.imsave(f'generated_images/{i}.jpeg', np.asarray((image + 1) / 2 * 255, dtype=np.uint8))
            
    stats = np.concatenate([np.array([np.round(np.sum(logs[:,0]) / 1000)]),np.round(np.mean(logs[1:], axis=0)),np.round(np.std(logs[1:], axis=0))])
    pd.DataFrame(stats).to_csv(f'results/hw_stats_batch_{INFERENCE_CFG["batch_size"]}_{experiment_path.split("/")[-1][:-4]}_{sys.argv[4]}.csv', index=None, header=None)




if __name__ == '__main__':
    perform_demo_inference()
    
    
    
def infer(checkpoint_path: str, exp_name: str):
    
    os.mkdir(f'generated_images/{exp_name}')
    
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2 if dataset.load_keypoints else None)
    )
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    model.load_state_dict(torch.load(Path(checkpoint_path)))
    model.eval()
    
    generated_im_num = 0
    print("Starting inference")
    for idx_batch, batch in tqdm(enumerate(dataloader)):
        print(idx_batch)
        data = batch
        kpts = data["keypoints"].to(device)
        kpts = rearrange(kpts.view(data['img'].shape[0], -1), "b c -> b c 1 1")
        model_fn = partial(model, x_self_cond=kpts)
        img2inpaint = data['img'].to(device) * data['mask'].to(device)
        
        samples = sample(model_fn, image_size=image_size, batch_size=kpts.shape[0], channels=channels, img2inpaint=img2inpaint)  # list of 1000 ndarrays of shape (batchsize, 3, 64, 64)
        for i in range(kpts.shape[0]):
            image = rearrange(samples[-1][i], 'c h w -> h w c')
            plt.imsave(f'generated_images/{exp_name}/{generated_im_num}.jpeg', np.asarray((image + 1) / 2 * 255, dtype=np.uint8))
            generated_im_num += 1
        
