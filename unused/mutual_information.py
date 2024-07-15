from mine_pytorch.mine.models.mine import MutualInformationEstimator
from run import Model
from pytorch_lightning import Trainer

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

import logging
logging.getLogger().setLevel(logging.ERROR)

class Latents(Dataset):
    def __init__(self, directory1, directory2, n_datapoints):
        self.files1 = sorted(glob.glob(os.path.join(directory1, '*_z.pth'))[:n_datapoints])
        self.files2 = sorted(glob.glob(os.path.join(directory2, '*_z.pth'))[:n_datapoints])

    def latent_shapes(self):
        return torch.load(self.files1[0]).flatten().squeeze().shape[0], torch.load(self.files2[0]).flatten().squeeze().shape[0]
    
    def __len__(self):
        return min(len(self.files1), len(self.files2))

    def __getitem__(self, idx):
        latent1 = torch.load(self.files1[idx])
        latent2 = torch.load(self.files2[idx])
        return latent1.squeeze().flatten(), latent2.squeeze().flatten()

class LatentFiles(Dataset):
    def __init__(self, latents1, latents2):
        # lists of 
        self.latents1 = latents1
        self.latents2 = latents2

    def latent_shapes(self):
        return self.latents1[0].flatten().squeeze().shape[0], self.latents2[0].flatten().squeeze().shape[0]
    
    def __len__(self):
        return min(len(self.latents1), len(self.latents2))

    def __getitem__(self, idx):
        latent1 = self.latents1[idx]
        latent2 = self.latents2[idx]
        return latent1.squeeze().flatten(), latent2.squeeze().flatten()
    
def calculate_post_map_mutual_information(from_, to_, map, dataset, n_datapoints=2000):
    to_files = [to_.decode_partial(torch.load(file).detach()) for file in sorted(glob.glob(f'latents/{to_.name}/{dataset}/*_z.pth'))[:n_datapoints]]
    from_files = [torch.load(file).detach() for file in sorted(glob.glob(f'latents/{from_.name}/{dataset}/*_z.pth'))[:n_datapoints]]
    mapped = [map(latent).flatten().squeeze().detach() for latent in from_files]
    print(len(mapped))
    print(mapped[0])
    dataset = LatentFiles(to_files, mapped)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    kwargs = {
        'lr': 1e-4,
        'batch_size': 100,
        'train_loader': dataloader,
        'test_loader': dataloader,
        'alpha': 1.0
    }

    latent_shapes = dataset.latent_shapes()
    model = MutualInformationEstimator(latent_shapes[0], latent_shapes[1], loss='mine', **kwargs).to('cpu')
    # max_epochs=200
    trainer = Trainer(max_epochs=50, logger=False)
    trainer.fit(model)

    test_mi = 0
    for i in range(100):
        test = model.test_step(next(iter(dataloader)), i)
        test_mi += test['test_mi'].item()

    print("MINE {}".format(test_mi/100))
    return test_mi/100

gan = Model('gan', 'gan', 'pickles/stylegan-celeba.pkl', 'celeba', 'w+')
def calculate_mutual_information_gan_latent_space(name_1, name_2, map_1, map_2, dataset, n_datapoints=2000):
    if map_1 is None:
        files_1 = [torch.load(file).detach() for file in sorted(glob.glob(f'latents/{name_1}/{dataset}/*_w+.pth')[:n_datapoints])]
        mapped_1 = [latent.detach() for latent in files_1]
    else:
        files_1 = [torch.load(file).detach() for file in sorted(glob.glob(f'latents/{name_1}/{dataset}/*_z.pth')[:n_datapoints])]
        mapped_1 = [map_1.predict(latent.flatten().detach().reshape(1, -1)) for latent in files_1]
    if map_2 is None:
        files_2 = [torch.load(file)[1].detach() for file in sorted(glob.glob(f'latents/{name_2}/{dataset}/*_w+.pth')[:n_datapoints])]
        mapped_2 = [latent.detach() for latent in files_2]
    else:
        files_2 = [torch.load(file).detach() for file in sorted(glob.glob(f'latents/{name_2}/{dataset}/*_z.pth')[:n_datapoints])]
        mapped_2 = [map_2.predict(latent.flatten().detach().reshape(1, -1)) for latent in files_2]

    dataset = LatentFiles(mapped_1, mapped_2)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    kwargs = {
        'lr': 1e-4,
        'batch_size': 100,
        'train_loader': dataloader,
        'test_loader': dataloader,
        'alpha': 1.0
    }

    model = MutualInformationEstimator(512, 512, loss='mine', **kwargs).to('cpu')
    # max_epochs=200
    trainer = Trainer(max_epochs=100, logger=False) # used 50 epochs for celeba experiment and 100 epochs for cifar
    trainer.fit(model)

    test_mi = 0
    for i in range(100):
        test = model.test_step(next(iter(dataloader)), i)
        test_mi += test['test_mi'].item()

    print("MINE {}".format(test_mi/100))
    return test_mi/100

def calculate_mutual_information(model_1, model_2, dataset, n_datapoints=2000):
    dataset = Latents(f'latents/{model_1}/{dataset}', f'latents/{model_2}/{dataset}', n_datapoints)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    kwargs = {
        'lr': 1e-4,
        'batch_size': 10,
        'train_loader': dataloader,
        'test_loader': dataloader,
        'alpha': 1.0
    }

    latent_shapes = dataset.latent_shapes()
    model = MutualInformationEstimator(latent_shapes[0], latent_shapes[1], loss='mine_biased', **kwargs).to('cpu')
    # max_epochs=200
    trainer = Trainer(max_epochs=50, logger=False)
    trainer.fit(model)

    test_mi = 0
    for i in range(100):
        test = model.test_step(next(iter(dataloader)), i)
        test_mi += test['test_mi'].item()

    print("MINE {}".format(test_mi/100))
    return test_mi/100


results = []
dataset = "celeba" # cifar
model_args = [
    ('vqvae', 'vqvae', 'z'),
    ('vae', 'vae-21', 'z'),
    ("nf", "nf-celeba", "z"),
    ('gan', 'gan', 'z'),
    ('gan', 'gan', 'w+'),
] 

# model_args = [
#     ('ae1', 'ae1', 'z'),
#     ('nf', 'nf', 'z'),
#     ('gan', 'gan', 'w+'),
# ]

for i in range(len(model_args)):
    for j in range(len(model_args)):
        model_1 = model_args[i]
        model_2 = model_args[j]
        if j < i:
            results.append([f"{model_1[0]} {model_1[2]}", f"{model_2[0]} {model_2[2]}", results[len(model_args) * j + i][2]])
        else:
            if model_1[0] == "gan" and model_1[2] == "w+":
                map_1 = None
            else:
                map_1 = load(f"pickles/latent_mapping_{model_1[1]}_gan_{dataset}_w+_linear.joblib")
            if model_2[0] == "gan" and model_2[2] == "w+":
                map_2 = None
            else:
                map_2 = load(f"pickles/latent_mapping_{model_2[1]}_gan_{dataset}_w+_linear.joblib")
            
            results.append([
                f"{model_1[0]} {model_1[2]}", 
                f"{model_2[0]} {model_2[2]}",
                calculate_mutual_information_gan_latent_space(model_1[1], model_2[1], map_1, map_2, dataset, 2000)
            ])
            

df = pd.DataFrame(results, columns=['Latent Space 1', 'Latent Space 2', 'MI'])
print(df)
reindexed = df.pivot(index='Latent Space 1', columns='Latent Space 2', values='MI')
sns.heatmap(reindexed, annot=True)

plt.savefig(f"plots/mutual_information_{dataset}.png")
plt.show()