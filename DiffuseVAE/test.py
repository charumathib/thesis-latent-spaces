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

from datasets import CIFAR10Dataset


if __name__ == '__main__':
    INPUT_RES = 64
    VAE_CKPT_PATH = "vae.ckpt" # todo: replace
    vae = VAE.load_from_checkpoint(
        VAE_CKPT_PATH,
        input_res=INPUT_RES,
    )
    vae.eval()
