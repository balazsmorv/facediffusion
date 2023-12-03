from schedulers import linear_beta_schedule
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from torch.optim import Adam
from model import Unet
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from functools import partial
from einops import rearrange
from ema_pytorch import EMA
from datetime import datetime
from ExpW_dataset import FacialExpressionsWithKeypointsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network_helper import extract
from inference import sample
import os

TRAIN_CFG = {
    # Model and train parameters
    'lr': 5e-5,
    'epochs': 10000,
    'timesteps': 1024,
    'channels': 3,

    # Dataset params
    #'dataset_pth': "/home/oem/Letöltések/Facialexp",
    'dataset_pth': "/Users/balazsmorvay/Downloads/FacialExpressionsTrainingData",
    'load_keypoints': True,
    'load_masks': True,
    'image_size': 64,
    'batch_size': 96,

    # Logging parameters
    'experiment_name': 'emotion_model_64_MBP',
    'eval_freq': 20,
    'save_and_sample_every': 10000,
    'model_checkpoint': None
    # '/home/jovyan/work/nas/USERS/tormaszabolcs/GIT/facediffusion/results/64x64_result_mask_part1/model_epoch_139.pth'
}

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", masks=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    if masks is not None:
        x_noisy = torch.where(masks == 0, x_noisy, x_start)

    predicted_noise = denoise_model(x_noisy, t)

    if masks is not None:
        predicted_noise = torch.where(masks == 0, predicted_noise, noise)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss



def reverse_transform_fun(t):
    t = (t + 1) / 2
    t = t.permute(1, 2, 0)  # CHW to HWC
    t = t * 255.
    t =  t.numpy().astype(np.uint8)
    return t

def img_transform_part(t):
    return t * 2 - 1

if __name__ == '__main__':

    # CONFIG SETUPS
    timesteps = TRAIN_CFG['timesteps']
    image_size = TRAIN_CFG['image_size']
    batch_size = TRAIN_CFG['batch_size']
    channels = TRAIN_CFG['channels']
    epochs = TRAIN_CFG['epochs']
    lr = TRAIN_CFG['lr']
    eval_freq = TRAIN_CFG['eval_freq']
    experiment_name = TRAIN_CFG['experiment_name']
    save_and_sample_every = TRAIN_CFG['save_and_sample_every']
    dataset_pth = TRAIN_CFG['dataset_pth']
    load_keypoints = TRAIN_CFG['load_keypoints']
    load_masks = TRAIN_CFG["load_masks"]
    writer = SummaryWriter()

    torch.manual_seed(0)

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas: torch.Tensor = 1. - betas
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
        Lambda(img_transform_part),
    ])

    mask_transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
    ])

    reverse_transform = Compose([Lambda(reverse_transform_fun)])

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    if torch.backends.mps.is_available():
        device = 'mps'
        print('MPS available')
    elif torch.cuda.is_available():
        device = 'cuda'
        print('CUDA available')
        print(torch.cuda.device_count())
    else:
        device = 'cpu'

    dataset = FacialExpressionsWithKeypointsDataset(csv_file=os.path.join(TRAIN_CFG['dataset_pth'],
                                                                          'labels_with_kpts.csv'),
                                                    root_dir=TRAIN_CFG['dataset_pth'],
                                                    img_transform=img_transform,
                                                    mask_transform=mask_transform,
                                                    imsize=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                            prefetch_factor=2, persistent_workers=True, pin_memory=True)

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim = 5 * 2 + 7 # 5 keypoints + emotion label one hot
    )
    model = torch.nn.DataParallel(model)
    if TRAIN_CFG['model_checkpoint'] is not None:
        model.load_state_dict(torch.load(TRAIN_CFG['model_checkpoint'], map_location='cpu'))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    ema = EMA(model, beta=0.9999, update_after_step=100, update_every=10)
    
    s = 0

    for epoch in range(epochs):
        model.train()

        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            s += 1
            optimizer.zero_grad()

            data = batch['image'].to(device)
            masks = batch['mask'].to(device)

            batch_size = data.shape[0]

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            keypoints = batch['keypoints'].to(device)
            emotion = batch['label'].to(device).float()
            condition = torch.cat((keypoints, emotion), dim=1)
            condition = rearrange(condition, 'b c -> b c 1 1')
            model_fn = partial(model, x_self_cond=condition)
            loss = p_losses(model_fn, data, t, loss_type="huber", masks=masks)

            if step % 100 == 0:
                pbar.set_description(f"Epoch {epoch+1}. Loss = {loss}")
                writer.add_scalar("loss", loss, s)

            loss.backward()
            optimizer.step()

            ema.update()

        if epoch % eval_freq == 0:
            with torch.inference_mode():
                try:
                    #torch.save(model.state_dict(),
                    #           Path("./results/" + experiment_name + "_epoch_" + str(epoch) + ".pth"))
                    torch.save(ema.ema_model.state_dict(),
                               Path("./results/" + experiment_name + "_epoch_" + str(epoch) + "ema.pth"))
                    ema.eval()
                    milestone = step // save_and_sample_every
                    batches = num_to_groups(4, batch_size)
                    # all_images_list = list(
                    #    map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels, img2inpaint=data*masks), batches))
                    all_images_list = sample(ema, image_size=image_size, batch_size=batch_size, channels=channels,
                                             img2inpaint=data * masks, device=device)
                    imlist = all_images_list  # [0]
                    lst = [torch.from_numpy(item) for item in imlist]
                    all_images = torch.cat(lst, dim=0)
                    all_images = (all_images + 1) * 0.5
                    writer.add_images("Images", all_images, epoch)
                    save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                except Exception as e:
                    print(e)

    # save model
    torch.save(ema.ema_model.state_dict(), Path("./results/" + experiment_name + "_final.pth"))

    writer.flush()
    writer.close()
