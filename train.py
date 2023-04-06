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
import torchvision


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
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


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
    experiment_name = "MNIST_model"
    torch.manual_seed(0)
    timesteps = 300
    image_size = 28

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
    save_and_sample_every = 1000
    channels = 1

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
    optimizer = Adam(model.parameters(), lr=1e-5)
    epochs = 1

    # dataset = FDF256Dataset(dirpath="/datadrive/FDF/dataset/train", load_keypoints=False, transform=transform)
    dataset = torchvision.datasets.MNIST('/datadrive/facediffusion/dataset', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                            ]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            batch = batch[0] # images
            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            # save generated images
            if step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
                imlist = all_images_list[0]
                lst = [torch.from_numpy(item) for item in imlist]
                all_images = torch.cat(lst, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)

        if epoch % 10 == 1:
            try:
                torch.save(model.state_dict(), Path("./results/" + experiment_name + "_epoch_" + str(epoch) + ".pth"))
            except Exception as e:
                print(e)

    # save model
    torch.save(model.state_dict(), Path("./results/" + experiment_name + ".pth"))
