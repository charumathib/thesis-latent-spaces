import cv2
import lpips

import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, ssim, fsim, issm, sre, sam, uiq
import torch
import PIL
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import glob
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, bundles

plt.rcParams.update(bundles.icml2024())

if __name__ == '__main__':
    # models with an encoder and decoder
    dataset = 'celeba'
    # dataset = 'cifar'

    models = ['random', 'vae-diffusion', 'vqvae', 'nf', 'dm']

    test_range = []
    if dataset == 'celeba':
        test_range = list(range(9001, 9101))
    else:
        for i in range(10):
            test_range += list(range((500 * i) + 401, (500 * i) + 411))

    results = []
    size = 32 if dataset == 'cifar' else 64

    if dataset == 'cifar':
        ground_truth_tensors_list = []
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for class_ in classes:
            for seed in range(401, 411):
                img = PIL.Image.open(f"data/cifar/{class_}/{str(seed).zfill(4)}.jpg")
                img = transforms.ToTensor()(img).unsqueeze(0) * 255
                ground_truth_tensors_list.append(img)
    else:
        ground_truth_tensors_list = [transforms.ToTensor()(PIL.Image.open(f"data/{dataset}/{str(idx).zfill(6)}.jpg")).unsqueeze(0) * 255 for idx in test_range]
        ground_truth_tensors_list = [
            transforms.CenterCrop(size)(transforms.Resize(size=size, antialias=True)(img))
            for img in ground_truth_tensors_list
        ]

        ground_truth_tensors_list_gen = [transforms.ToTensor()(PIL.Image.open(f"data/{dataset}_gen/{str(idx).zfill(6)}.jpg")).unsqueeze(0) * 255 for idx in test_range]
        ground_truth_tensors_list_gen = [
            transforms.CenterCrop(size)(transforms.Resize(size=size, antialias=True)(img))
            for img in ground_truth_tensors_list_gen
        ]

    ground_truth_tensors = torch.cat(ground_truth_tensors_list, 0).to(torch.uint8)
    ground_truth_tensors_gen = torch.cat(ground_truth_tensors_list_gen, 0).to(torch.uint8)

    # decoder only models (GAN)
    for model in models:
        new_tensors_list = [transforms.ToTensor()(PIL.Image.open(f"mapped/{dataset}/{str(idx).zfill(6)}_gen_{model}_to_gan_linear.jpg")).unsqueeze(0) * 255 for idx in test_range]
        new_tensors = torch.cat(new_tensors_list, 0).to(torch.uint8)
        # print(new_tensors.shape)

        sys.stdout = open(os.devnull, 'w')
        loss_fn = lpips.LPIPS(net='alex')
        # normalize to [-1, 1]
        d = loss_fn.forward((new_tensors / 127.5) - 1, (ground_truth_tensors_gen / 127.5) - 1).squeeze()

        fid = FrechetInceptionDistance(feature=64)
        fid.update(ground_truth_tensors_gen, real=True)
        fid.update(new_tensors, real=False)
        sys.stdout = sys.__stdout__

        e_name = model.split('-')[0] if '-' in model else 'ae' if model == 'ae1' else model

        results.append(
            {
                "Encoder": e_name,
                "Decoder": "gan",
                "FID": fid.compute().item(),
                "LPIPS": d.mean().item(),
                "RMSE": rmse(ground_truth_tensors.numpy(), new_tensors.numpy())
            }
        )

    # all encoder - decoder models
    for i in range(len(models)):
        for j in range(len(models)):
            if models[j] == 'random': continue
            if i == j:
                # change for celeba
                new_tensors_list = [transforms.ToTensor()(PIL.Image.open(f"mapped/{dataset}/{str(idx).zfill(6)}_{models[i] if models[i] != 'vae-diffusion' else 'vae'}.jpg")).unsqueeze(0) * 255 for idx in test_range]
            else:
                new_tensors_list = [transforms.ToTensor()(PIL.Image.open(f"mapped/{dataset}/{str(idx).zfill(6)}_{models[i]}_to_{models[j] if models[j] != 'vae-diffusion' else 'vae'}_linear.jpg")).unsqueeze(0) * 255 for idx in test_range]

            new_tensors = torch.cat(new_tensors_list, 0).to(torch.uint8)
            # print(new_tensors.shape)

            sys.stdout = open(os.devnull, 'w')
            loss_fn = lpips.LPIPS(net='alex')
            # normalize to [-1, 1]
            d = loss_fn.forward((new_tensors / 127.5) - 1, (ground_truth_tensors / 127.5) - 1).squeeze()

            fid = FrechetInceptionDistance(feature=64)
            fid.update(ground_truth_tensors, real=True)
            fid.update(new_tensors, real=False)
            sys.stdout = sys.__stdout__
            
            e_name = models[i].split('-')[0] if '-' in models[i] else 'ae' if models[i] == 'ae1' else models[i]
            d_name = models[j].split('-')[0] if '-' in models[j] else 'ae' if models[j] == 'ae1' else models[j]

            results.append(
                {
                    "Encoder": e_name,
                    "Decoder": d_name,
                    "FID": fid.compute().item(),
                    "LPIPS": d.mean().item(),
                    "RMSE": rmse(ground_truth_tensors.numpy(), new_tensors.numpy())
                }
            )

    df = pd.DataFrame(results)
    df.to_pickle("celeba_img_results.pkl")

    df = pd.read_pickle("celeba_img_results.pkl").dropna()
    df['Encoder'] = df['Encoder'].str.upper()
    df['Decoder'] = df['Decoder'].str.upper()

    df = df.replace("RANDOM", "Rand.")

    df.Encoder=pd.Categorical(df.Encoder,categories=df.Encoder.unique(),ordered=True)
    df.Decoder=pd.Categorical(df.Decoder,categories=df.Decoder.unique(),ordered=True)
    print(df)
    plt.figure(figsize=(2.0, 1.7))
    sns.heatmap(df.pivot(index="Decoder", columns="Encoder", values="FID"), annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title("FID")
    plt.savefig(f"plots/fid_heatmap_{dataset}.png", dpi=300)

    plt.figure(figsize=(2.0, 1.7))
    sns.heatmap(df.pivot(index="Decoder", columns="Encoder", values="LPIPS"), annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title("LPIPS")
    plt.savefig(f"plots/lpips_heatmap_{dataset}.png", dpi=300)

    plt.figure(figsize=(2.0, 1.7))
    sns.heatmap(df.pivot(index="Decoder", columns="Encoder", values="RMSE"), annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title("RMSE")
    plt.savefig(f"plots/rmse_heatmap_{dataset}.png", dpi=300)
