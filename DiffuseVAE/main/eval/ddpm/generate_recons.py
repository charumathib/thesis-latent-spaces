# Helper script to generate reconstructions from a conditional DDPM model
# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import copy

import hydra
import pytorch_lightning as pl
import torch
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device, get_dataset
import numpy as np

from datasets import CIFAR10Dataset
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

# latent space dim is 512 for 64x64 images and 1024 for 256x256 images
@hydra.main(config_path=os.path.join(p, "configs"))
def generate_recons(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    decoder_args = {
        'in_channels': config_ddpm.data.n_channels,
        'model_channels': config_ddpm.model.dim,
        'out_channels': 3,
        'num_res_blocks': config_ddpm.model.n_residual,
        'attention_resolutions': attn_resolutions,
        'channel_mult': dim_mults,
        'use_checkpoint': False,
        'dropout': config_ddpm.model.dropout,
        'num_heads': config_ddpm.model.n_heads,
        'z_dim': config_ddpm.evaluation.z_dim,
        'use_scale_shift_norm': config_ddpm.evaluation.z_cond,
        'use_z': config_ddpm.evaluation.z_cond
    }
    print(decoder_args)

    decoder = SuperResModel(**decoder_args)

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM

    online_ddpm_args = {
        'beta_1': config_ddpm.model.beta1,
        'beta_2': config_ddpm.model.beta2,
        'T': config_ddpm.model.n_timesteps,
        'var_type': config_ddpm.evaluation.variance,
    }
    print(online_ddpm_args)

    online_ddpm = ddpm_cls(decoder, **online_ddpm_args)

    target_ddpm_args = {
        "beta_1": config_ddpm.model.beta1,
        "beta_2": config_ddpm.model.beta2,
        "T": config_ddpm.model.n_timesteps,
        "var_type": config_ddpm.evaluation.variance
    }

    target_ddpm = ddpm_cls(ema_decoder, **target_ddpm_args)

    ddpm_wrapper_args = {
        'conditional': True,
        'pred_steps': n_steps,
        'eval_mode': "recons",
        'resample_strategy': config_ddpm.evaluation.resample_strategy,
        'skip_strategy': config_ddpm.evaluation.skip_strategy,
        'sample_method': config_ddpm.evaluation.sample_method,
        'sample_from': config_ddpm.evaluation.sample_from,
        'data_norm': config_ddpm.data.norm,
        'temp': config_ddpm.evaluation.temp,
        'guidance_weight': config_ddpm.evaluation.guidance_weight,
        'z_cond': config_ddpm.evaluation.z_cond,
        'ddpm_latents': ddpm_latents,
        'strict': True,
    }

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        **ddpm_wrapper_args
    )

    image_size = config_ddpm.data.image_size

    # # load PIL image
    # image = Image.open('/Users/charumathibadrinath/thesis-latent-spaces/data/cifar_gen/0001.jpg')
    # image = (np.asarray(image).astype(np.float) / 127.5) - 1.0
    # image = torch.from_numpy(image).permute(2, 0, 1).float()

    # # encode the image
    # with torch.no_grad():
    #     mu, logvar = vae.encode(image * 0.5 + 0.5)
    #     z = vae.reparameterize(mu, logvar).unsqueeze(0)
    
    # print(z.shape)

    # # reconstruct from z

    for i in tqdm(range(9001, 9101)):
        z = torch.load(f"/Users/charumathibadrinath/thesis-latent-spaces/latents/vae-diffusion/celeba_real/{str(i).zfill(6)}.pth").flatten().detach().unsqueeze(0)
        with torch.no_grad():
            recons = vae.decode(z.view(1, 512, 1, 1))
            diff_recon, vae_recon = ddpm_wrapper.predict_step(recons, 0, recons, None)
            
            img_ = (diff_recon['20'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(f"/Users/charumathibadrinath/thesis-latent-spaces/mapped/celeba/{str(i).zfill(6)}vae-diffusion.jpg")

            # save vae recon
            img_ = (vae_recon.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(f"/Users/charumathibadrinath/thesis-latent-spaces/mapped/celeba/{str(i).zfill(6)}vae.jpg")


if __name__ == "__main__":
    generate_recons()
