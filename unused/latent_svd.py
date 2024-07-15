import torch
import numpy as np


# models = ['vae-21', 'vqvae', 'gan', 'nf-celeba']
models = ['ae1', 'nf', 'gan']

def n(name):
    if name == "ae1":
        return "ae"
    elif name == 'vae-21':
        return 'vae'
    elif name == 'nf-celeba':
        return 'nf'
    else:
        return name

dataset = 'cifar' # 'celeba'

for model in models:
    latents = [torch.load(f"latents/{model}/{dataset}/{idx}_{'w+' if model == 'gan' else 'z'}.pth").squeeze().flatten().detach() for idx in range(500)]
    latents = torch.stack(latents, -1)
    latentsTlatents = (latents.T @ latents).detach()

    singular_values = torch.linalg.svdvals(latentsTlatents)**2
    singular_values = np.cumsum(singular_values)/torch.sum(singular_values)
    print(model, np.argmax(singular_values > 0.9).item())