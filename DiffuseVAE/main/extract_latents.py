import click
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from util import configure_device, get_dataset
from models.vae import VAE
from PIL import Image
import torchvision.transforms as T

@click.group()
def cli():
    pass


@cli.command()
# @click.option("--vae-chkpt-path", default="~/Downloads/vae_celeba64_loss=0.0000.ckpt")
@click.option("--vae-chkpt-path", default="~/Downloads/vae_cifar10_loss=0.00.ckpt")
@click.option("--root", default="/Users/charumathibadrinath/thesis-latent-spaces/data")
@click.option("--device", default="gpu:0")
@click.option("--dataset-name", default="cifar10")
@click.option("--image-size", default=32)
@click.option("--save-path", default=os.getcwd())
def extract(
    vae_chkpt_path,
    root,
    device="cpu",
    dataset_name="cifar10",
    image_size=32,
    save_path=os.getcwd(),
):
    # dev, _ = configure_device(device)
    dev = 'cpu'

    # # Dataset
    # dataset = get_dataset(dataset_name, root, image_size, norm=False, flip=False)
    # print(dataset)

    # # Loader
    # loader = DataLoader(
    #     dataset,
    #     1,
    #     num_workers=1,
    #     pin_memory=True,
    #     shuffle=False,
    #     drop_last=False,
    # )

    # Load VAE
    vae = VAE.load_from_checkpoint(vae_chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    with torch.no_grad():
        #for ind, class_ in enumerate(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]):
        for i in tqdm(range(1, 5001)):
                img = Image.open(f"/Users/charumathibadrinath/thesis-latent-spaces/data/cifar_gen/{str(i).zfill(4)}.jpg")
                # img = Image.open(f"/Users/charumathibadrinath/thesis-latent-spaces/data/cifar/{class_}/{str(i).zfill(4)}.jpg")
                # img = Image.open(f"/Users/charumathibadrinath/Diffusion-GAN/diffusion-stylegan2/out/{str(i).zfill(6)}.jpg")
                img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                #img = T.Resize(64)(img)
                mu, logvar = vae.encode(img * 0.5 + 0.5)
                z = vae.reparameterize(mu, logvar).unsqueeze(0)
                # torch.save(z, f"./latents_gen/{str(i).zfill(6)}_z.pth")
                torch.save(z, f"/Users/charumathibadrinath/thesis-latent-spaces/latents/vae-diffusion/cifar_gen/{str(i).zfill(4)}_z.pth") # (ind + 1)

if __name__ == "__main__":
    cli()
