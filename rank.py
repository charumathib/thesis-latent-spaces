import glob
import torch
import numpy as np
from run import Model, OneLayerNN
import matplotlib.pyplot as plt


def map_svd(from_, to_, dataset, pickle_location):
    map = torch.load(f"pickles/{pickle_location}.pth")

    weights = map.weight
    singular_values = torch.linalg.svdvals(weights)

    # Calculate proportion of variance explained
    variance_explained = singular_values / torch.sum(singular_values)

    # Calculate cumulative variance explained
    cumulative_variance_explained = np.cumsum(variance_explained.detach())

    # Create plot
    plt.figure(figsize=(6, 4))
    plt.plot(cumulative_variance_explained)
    plt.axhline(y=0.99, color='purple', linestyle='--', label='0.99')  # Add horizontal line at 0.95
    plt.axhline(y=0.95, color='r', linestyle='--'m label='0.95')  # Add horizontal line at 0.95
    plt.axhline(y=0.8, color='g', linestyle='--', label='0.8')  # Add horizontal line at 0.8
    plt.legend()

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'SVD of {from_} {to_} linear mapping on {dataset}')
    plt.savefig(f"plots/svd_{from_}_{to_}_{dataset}.png")

map_svd('gan', 'vae', 'celeba', 'gan_to_vae')