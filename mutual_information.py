from mine_pytorch.mine.models.mine import MutualInformationEstimator
from run import Model
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

import logging
logging.getLogger().setLevel(logging.ERROR)

class Latents(Dataset):
    def __init__(self, directory1, directory2):
        self.files1 = glob.glob(os.path.join(directory1, '*_z.pth'))
        self.files2 = glob.glob(os.path.join(directory2, '*_z.pth'))

    def __len__(self):
        return min(len(self.files1), len(self.files2))

    def __getitem__(self, idx):
        latent1 = torch.load(self.files1[idx])
        latent2 = torch.load(self.files2[idx])
        return latent1.squeeze(), latent2.squeeze()

def calculate_mutual_information():
    dataset = Latents('latents/gan/celeba', 'latents/vae/celeba')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    kwargs = {
        'lr': 1e-4,
        'batch_size': 100,
        'train_loader': dataloader,
        'test_loader': dataloader,
        'alpha': 1.0
    }
    model = MutualInformationEstimator(512, 128, loss='mine_biased', **kwargs).to('cpu')
    # max_epochs=200
    trainer = Trainer(max_epochs=50, logger=False)
    trainer.fit(model)
    print(model.test_step(next(iter(dataloader)), 0))
    print(model.test_step(next(iter(dataloader)), 1))
    print(model.test_step(next(iter(dataloader)), 2))

    # print("MINE {}".format(model.avg_test_mi))


calculate_mutual_information()