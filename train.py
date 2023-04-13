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
from fdh256_dataset import FDF256Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network_helper import extract
from inference import sample


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


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


if __name__ == '__main__':

    writer = SummaryWriter()

    experiment_name = "model"
    torch.manual_seed(0)
    timesteps = 300
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

    transform = Compose([
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Resize(image_size),
        CenterCrop(image_size),
        Lambda(lambda t: (t * 2) - 1),
    ])

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 10000
    channels = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model = torch.nn.DataParallel(model)
    model.to(device)
    lr = 1e-5
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = 1
    batch_size = 512

    dataset = FDF256Dataset(dirpath="/datadrive/FDF/dataset/train", load_keypoints=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            writer.add_scalar("loss", loss, epoch)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 1:
                try:
                    torch.save(model.state_dict(),
                               Path("./results/" + experiment_name + "_epoch_" + str(epoch) + ".pth"))
                    milestone = step // save_and_sample_every
                    batches = num_to_groups(4, batch_size)
                    all_images_list = list(
                        map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
                    imlist = all_images_list[0]
                    lst = [torch.from_numpy(item) for item in imlist]
                    all_images = torch.cat(lst, dim=0)
                    all_images = (all_images + 1) * 0.5
                    writer.add_images("Images", all_images, epoch)
                    save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                except Exception as e:
                    print(e)

    # save model
    torch.save(model.state_dict(), Path("./results/" + experiment_name + ".pth"))

    writer.flush()
    writer.close()
