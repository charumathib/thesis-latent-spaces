import os
import glob
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from run import Model, OneLayerNN
import umap

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
def plot_images(method, img_dir, train_size, test_size, dim=2, save=True):
    """
    Plot the images in a directory using a dimensionality reduction technique.
    """
    print(f"Plotting {img_dir}...")

    # Load the data
    files = glob.glob(f'{img_dir}/*.jpg')
    train_data = np.array([plt.imread(file).flatten() for file in files][:train_size])
    test_data = np.array([plt.imread(file).flatten() for file in files][train_size:train_size + test_size])

    print(f"train dataset shape {train_data.shape}")
    print(f"test dataset shape {test_data.shape}")

    # Perform dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=dim, perplexity=50, method='exact' if dim > 4 else 'barnes_hut', random_state=7)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=dim, random_state=7)
    elif method == 'pca':
        reducer = PCA(n_components=dim)
    else:
        raise ValueError("dim_reduction must be one of 'tsne', 'umap', or 'pca'")

    train_data_compressed = reducer.fit_transform(train_data)
    # train_data_compressed = fit.transform(train_data)
    # test_data_compressed = fit.transform(test_data)

    if dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(train_data[:, 0], train_data[:, 1])
        plt.savefig(f"plots/{method}_celeba_2d.png")

    if save:
        print(f"saving train to: pickles/celeba_train_{train_size}_datapoints_{dim}d_{method}.pth")
        # print(f"saving test to: pickles/celeba_test_{test_size}_datapoints_{dim}d_{method}.pth")
        torch.save(train_data_compressed, f"pickles/celeba_train_{train_size}_datapoints_{dim}d_{method}.pth")
        # torch.save(test_data_compressed, f"pickles/celeba_test_{test_size}_datapoints_{dim}d_{method}.pth")

def plot_latents(dim_reduction, name, pickle_dir, decode_partial=False, map=None, annotate=[], model=None):
    """
    Plot the latent space of a model using a dimensionality reduction technique.
    """
    print(f"Plotting {pickle_dir}...")

    # Load the data
    files = glob.glob(f'{pickle_dir}/*_z.pth')
    if decode_partial:
        data = [model.decode_partial(torch.load(file)).flatten().squeeze() for file in files]
    else:
        data = [torch.load(file).squeeze() for file in files]
    
    data = data[:2000]
    if map is not None:
        # print(map)
        data = [map(tensor) for tensor in data]

    data = np.array([tensor.detach().numpy() for tensor in data])
    print(data.shape)

    # Perform dimensionality reduction
    if dim_reduction == 'tsne':
        reducer = TSNE(n_components=2, perplexity=50, random_state=7)
    elif dim_reduction == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=7)
    elif dim_reduction == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("dim_reduction must be one of 'tsne', 'umap', or 'pca'")

    data_2d = reducer.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1])
    
    if annotate:
        for i, point in enumerate(annotate):
            plt.scatter(data_2d[point][0], data_2d[point][1], color=colors[i])
            plt.annotate(point, (data_2d[point][0], data_2d[point][1]))

    plt.show()
    # plt.savefig(f"plots/{name}.png")

# plot_latents('tsne', "vae", pickle_dir="latents/vae/celeba", decode_partial=False, map=None, annotate=range(0, 2000, 200))
# plot_latents('tsne', "nf", pickle_dir="latents/nf-celeba/celeba", decode_partial=False, map=None, annotate=range(0, 2000, 200))
# plot_latents('tsne', "vae_to_gan_z", pickle_dir="latents/vae-nonlinear/celeba", decode_partial=False, map=torch.load(f"pickles/latent_mapping_vae_gan_celeba_z.pth"), annotate=range(0, 2000, 200))
# plot_latents('tsne', "gan_z", pickle_dir="latents/gan/celeba", decode_partial=False, map=None, annotate=range(0, 2000, 200), model=Model("gan", "gan", "pickles/stylegan-celeba.pkl", "celeba", "w+"))

# we want to see if the mapped version shares the same general distribution as the unmapped version / 
# general characteristics of the latent space
# annotate some proportion of datapoints and look at their relative positions in both plots
    
plot_images('tsne', 'gen/gan/celeba', 128, 10, 128, save=True)