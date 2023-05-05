from copy import deepcopy

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

from fdh256_dataset import FDF256Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network_helper import extract
from inference import sample

TRAIN_CFG = {
    # Model and train parameters
    'lr': 5e-5,
    'epochs': 10,
    'timesteps': 1024,
    'channels': 3,

    # Dataset params
    'dataset_pth': "/home/jovyan/work/nas/USERS/tormaszabolcs/DATA/FDF256/FDF/data/train",
    'load_keypoints': True,
    'load_masks': True,
    'image_size': 64,
    'batch_size': 16,

    # Logging parameters
    'experiment_name': 'distilled_model',
    'eval_freq': 10,
    'save_and_sample_every': 10000,
    'model_checkpoint': '/home/jovyan/work/nas/USERS/tormaszabolcs/GIT/facediffusion/results/64x64_result_mask_part2/model_epoch_399.pth'
}


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


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


# forward diffusion (using the nice property)
def q_sample(schedule, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(schedule, denoise_model, x_start, t, noise=None, loss_type="l1", masks=None, teacher_model=None,
             teacher_schedule=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(schedule, x_start=x_start, t=t, noise=noise)

    if masks is not None:
        x_noisy = torch.where(masks == 0, x_noisy, x_start)

    predicted_noise = denoise_model(x_noisy, t)

    if teacher_model is not None:
        # Two steps by the teacher
        teacher_pred_x_prev = ddim_sample(teacher_schedule, teacher_model, x_noisy, 2 * t + 1)
        if masks is not None:
            teacher_pred_x_prev = torch.where(masks == 0, teacher_pred_x_prev, x_start)
        with torch.no_grad():
            teacher_pred_noise = teacher_model(teacher_pred_x_prev, 2 * t)

        # Cuz if t == 0, teacher not provides the wanted information, the last step is t == 1 -> 2t == 2
        # this will be the target for the student model
        t_shaped = t.view(t.shape[0], 1, 1, 1)
        noise = torch.where(t_shaped != 0, teacher_pred_noise, noise)

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


if __name__ == '__main__':
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
    ])

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    dataset = FDF256Dataset(dirpath=dataset_pth, load_keypoints=load_keypoints, img_transform=img_transform,
                            load_masks=load_masks, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                            prefetch_factor=2, persistent_workers=True, pin_memory=True)

    teacher_schedule = Schedule(TRAIN_CFG['timesteps'])
    teacher = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        self_condition_dim=(7 * 2 if dataset.load_keypoints else None)
    )
    teacher = torch.nn.DataParallel(teacher)
    if TRAIN_CFG['model_checkpoint'] is not None:
        teacher.load_state_dict(torch.load(TRAIN_CFG['model_checkpoint'], map_location='cpu'))
    teacher.to(device)
    teacher.eval()

    student = deepcopy(teacher)
    student.to(device)

    opt_func = partial(Adam)
    optimizer = opt_func(student.parameters(), lr=lr)

    ema = EMA(student, beta=0.999, update_after_step=1, update_every=1)

    current_infer_steps = TRAIN_CFG['timesteps'] // 2

    stages_done = 0
    epochs_done = 0
    while current_infer_steps > 0:
        teacher.eval()
        student_schedule = Schedule(current_infer_steps)

        stage_epochs = epochs if current_infer_steps > 2 else 2 * epochs
        for epoch in range(stage_epochs):
            epoch_loss = 0.
            student.train()

            with tqdm(total=len(dataloader), leave=True) as pbar:
                pbar.update(1)
                for step, batch in enumerate(dataloader):
                    optimizer.zero_grad()

                    if not dataset.load_keypoints:
                        data = batch.to(device)
                    else:
                        data = batch['img'].to(device)

                    if dataset.load_masks:
                        masks = batch['mask'].to(device)
                    else:
                        masks = None

                    batch_size = data.shape[0]

                    # Algorithm 1 line 3: sample t uniformally for every example in the batch
                    t = torch.randint(0, student_schedule.steps, (batch_size,), device=device).long()

                    if not dataset.load_keypoints:
                        model_fn = student
                        teacher_fn = teacher
                    else:
                        keypoints = batch['keypoints'].to(device)
                        keypoints = rearrange(keypoints.view(batch_size, -1), "b c -> b c 1 1")
                        model_fn = partial(student, x_self_cond=keypoints)
                        teacher_fn = partial(teacher, x_self_cond=keypoints)

                    loss = p_losses(student_schedule, model_fn, data, t, loss_type="huber", masks=masks,
                                    teacher_model=teacher_fn, teacher_schedule=teacher_schedule)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    if ((step + 1) % 100 == 0) or ((step + 100) >= len(dataloader)):
                        pbar.set_postfix({
                            "AvgLoss": str(epoch_loss / (step + 1)),
                            "Epoch": f"{epoch + 1}/{stage_epochs}",
                            "InferStage": str(current_infer_steps),
                            "LR": str(lr)
                        })
                        pbar.update(100)

                    writer.add_scalar("loss", loss.item(), epochs_done * len(dataloader) + step)

                    ema.update()

                if (epochs_done + 1) % eval_freq == 0:
                    with torch.no_grad():
                        try:
                            torch.save(student.state_dict(),
                                       Path("./results/" + experiment_name + "_epoch_" + str(epochs_done) + ".pth"))
                            torch.save(ema.ema_model.state_dict(),
                                       Path("./results/" + experiment_name + "_epoch_" + str(epochs_done) + "_ema.pth"))
                            ema.eval()
                            milestone = step // save_and_sample_every
                            batches = num_to_groups(4, batch_size)
                            # all_images_list = list(
                            #    map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels, img2inpaint=data*masks), batches))
                            all_images_list = sample(ema, image_size=image_size, batch_size=batch_size,
                                                     channels=channels,
                                                     img2inpaint=data * masks)
                            imlist = all_images_list  # [0]
                            lst = [torch.from_numpy(item) for item in imlist]
                            all_images = torch.cat(lst, dim=0)
                            all_images = (all_images + 1) * 0.5
                            writer.add_images("Images", all_images, epochs_done)
                            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
                        except Exception as e:
                            print(e)

                epochs_done += 1

        current_infer_steps = current_infer_steps // 2

        teacher = deepcopy(student)
        teacher_schedule = deepcopy(student_schedule)

        lr -= (TRAIN_CFG['lr'] / int(np.log2(TRAIN_CFG['timesteps'])))
        optimizer = opt_func(student.parameters(), lr=lr)

    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    print(lrs)

    # save model
    torch.save(student.state_dict(), Path("./results/" + experiment_name + ".pth"))
    torch.save(ema.ema_model.state_dict(), Path("./results/" + experiment_name + "_ema.pth"))

    writer.flush()
    writer.close()
