import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import sys

import pandas as pd

import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms

from cifar_ae import create_model

import legacy

import json

from glow.datasets import postprocess, preprocess
from glow.model import Glow

device = torch.device("cpu")

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

class Model():
    def __init__(self, name, pickle, dataset, latent_layer="z"):
        self.name = name
        self.dataset = dataset
        self.model = None
        self.pixel_shape = None
        self.z_shape = None
        self.z_dim = None
        self.latent_shape = None
        self.latent_dim = None
        self.latent_layer = latent_layer
        
        # temporarily redirect stdout to devnull to suppress printing of state dict keys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        if self.name == "nf": # normalizing flow
            with open("glow/hparams.json") as json_file:  
                hparams = json.load(json_file)
                
            self.model_ = Glow((32, 32, 3), hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                        hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], 10,
                        hparams['learn_top'], hparams['y_condition'])

            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
            self.model_.set_actnorm_init()
            self.model_ = self.model_.eval()
            sys.stdout = original_stdout

            assert self.dataset == "cifar"
            self.z_shape = self.latent_shape = (1, 48, 4, 4)
            self.z_dim = self.latent_dim = 48 * 4 * 4
        elif self.name == "ae":
            self.model_ = create_model()
            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
            sys.stdout = original_stdout

            assert self.dataset == "cifar"
            self.z_shape = self.latent_shape = (1, 48, 4, 4)
            self.z_dim = self.latent_dim = 48 * 4 * 4
        elif self.name == "vae":
            self.model_ = torch.load(pickle, map_location='cpu')
            sys.stdout = original_stdout
            assert self.dataset == "celeba"

            self.z_shape = self.latent_shape = (1, 128)
            self.z_dim = self.latent_dim = 128
        elif self.name == "gan":
            with open(pickle, 'rb') as f:
                self.model_ = legacy.load_network_pkl(f)['G_ema'].to(device)
            sys.stdout = original_stdout
            assert self.dataset == "cifar" or self.dataset == "celeba" or self.dataset == "metfaces"

            self.z_shape = self.latent_shape = (1, 512)
            self.z_dim = self.latent_dim = 512

            if self.latent_layer == "w+":
                if self.dataset == "cifar":
                    self.latent_shape = (1, 8, 512)
                    self.latent_dim = 8 * 512
                elif self.dataset == "celeba":
                    self.latent_shape = (1, 14, 512)
                    self.latent_dim = 14 * 512
                elif self.dataset == "metfaces":
                    self.latent_shape = (1, 18, 512)
                    self.latent_dim = 18 * 512
        
        if self.dataset == "cifar":
            self.pixel_shape = (1, 3, 32, 32)
        elif self.dataset == "celeba" or self.dataset == "metfaces":
            self.pixel_shape = (1, 3, 150, 150)

        if self.model_ is None: raise NotImplementedError
        
        print(f">> ###### Loaded {self.name} model ###### ")
        print(f">> Dataset: {self.dataset}")
        print(f">> Pixel shape: {self.pixel_shape}")
        print(f">> z shape: {self.z_shape}")
        print(f">> Latent layer: {self.latent_layer}")
        print(f">> Latent shape: {self.latent_shape}")

    def encode(self, img):
        if self.name == "nf":
            z, _, _ = self.model_(preprocess(img))
            return z
        if self.name == "ae":
            return self.model_.encoder(img)
        elif self.name == "vae":
            mu, log_var = self.model_.encode(img)
            return self.model_.reparameterize(mu, log_var)
        else:
            raise NotImplementedError(f">> Encode not implemented for {self.name}")
    
    def decode_partial(self, z):
        if self.latent_layer == "z":
            return z
        elif self.latent_layer == "w+":
            return self.model_.mapping(z, None)

        raise NotImplementedError

    def decode(self, latent):
        """
        Decodes the given latent vector into an image.

        Args:
            latent (torch.Tensor): The latent vector to decode into an image.
        Returns:
            torch.Tensor: The decoded image in pixel space.
        """
        latent = latent.view(self.z_shape if self.latent_layer == "z" else self.latent_shape)

        if self.name == "nf":
            return postprocess(self.model_(y_onehot=None, z=latent, temperature=0, reverse=True)).cpu()
        elif self.name == "ae":
            return self.model_.decoder(latent)
        elif self.name == "vae":
            return self.model_.decode(latent).view(self.pixel_shape)
        elif self.name == "gan":
            if self.latent_layer == "z":
                img = self.model_(latent, None, noise_mode='none', force_fp32=True)
            elif self.latent_layer == "w+":
                img = self.model_.synthesis(latent, noise_mode='none', force_fp32=True)
            
            if self.pixel_shape[2] != img.shape[2]:
                img = transforms.Resize(size=self.pixel_shape[2], antialias=True)(img)

            return img
        else:
            raise NotImplementedError
    
    def reconstruct(self, img, filename, n_recon=5) :
        if self.name == "nf":
            z = self.encode(img)
            pic = self.decode(z).squeeze()
            plt.imshow(pic.permute(1,2,0))
            plt.savefig(f"rec/{filename}")
        elif self.name == "ae":        
            pics = img
            for _ in range(n_recon):
                pic = self.model_(img)[1]
                pics = torch.cat((pics, recon), dim=0)
                save_image(pic, f"rec/{filename}")
        elif self.name == "vae":
            pics = img
            for _ in range(n_recon):
                recon, _, _ = self.model_(img)
                img_size = self.pixel_shape
                pic = recon[0].view(1, 3, img_size, img_size)
                pics = torch.cat((pics, pic), dim=0)
                save_image(pic, f"rec/{filename}")
        else:
            raise NotImplementedError
        
    def save_image(self, img, filename):
        if self.name == "nf":
            plt.imshow(img.detach().squeeze().permute(1,2,0))
            # hide axes
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.savefig(f"{filename}")
        elif self.name == "ae" or self.name == "vae":
            save_image(img, filename)
        else:
            img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(filename)

    def generate(self, seed):
        if self.name == "nf":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            img = postprocess(self.model_(y_onehot=None, z=latent, temperature=1, reverse=True))
        elif self.name == "ae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            img = self.model_.decoder(latent)
        elif self.name == "vae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape(self.dataset))).to(torch.float).to(device)
            img = self.model_.decode(latent).view(self.pixel_shape)
        elif self.name == "gan":
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.model_.z_dim)).to(torch.float32).to(device)
            latent = self.model_.mapping(z, None)
            img = self.model_(z, None, noise_mode='none', force_fp32=True)

            if self.pixel_shape[2] != img.shape[2]:
                img = transforms.Resize(size=self.pixel_shape[2], antialias=True)(img)
        
        self.save_image(img, f'gen/{self.name}/{self.dataset}/{str(seed).zfill(6)}.jpg')

        return z, img

def train_latent_space_mapping(
        from_, 
        to_, 
        data, 
        n_epochs=10, 
        n_datapoints=2000,
        criterion=nn.MSELoss(),
        optimizer="sgd",
        learning_rate=1e-3,
        momentum=None,
        batch_size=100, 
        from_saved=False, 
        save_pickles=True, 
        plot_losses=True,
        load_gen_img_from_file=False
    ):
    print(f">> Training latent space mapping from {from_.name} to {to_.name} over {n_datapoints} datapoints and {n_epochs} epochs")
    
    map = nn.Linear(from_.latent_dim, to_.latent_dim)

    if optimizer == "adam":
        optimizer = optim.Adam(map.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(map.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise NotImplementedError

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for iter in range(n_datapoints):
            if save_pickles and (from_saved or epoch > 0):
                from_latent = torch.load(f"latents/{from_.name}/{from_.dataset}/{iter}_z.pth")
                to_latent = torch.load(f"latents/{to_.name}/{to_.dataset}/{iter}{'_z' if to_.dataset != 'metfaces' else ''}.pth")
                if to_.latent_layer == "w+":
                    to_latent = to_.decode_partial(to_latent)
            else:
                # use GAN to generate the data
                if data == None:
                    if from_.name == "gan":
                        from_latent, img = from_.generate(iter)
                        if load_gen_img_from_file:
                            img = PIL.Image.open(f"gen/{to_.name}/{to_.dataset}/{str(iter).zfill(6)}.jpg").convert('RGB')
                            img = transforms.ToTensor()(img).unsqueeze(0)
                        to_latent = to_.encode(img)
                    elif to_.name == "gan":
                        to_latent, img = to_.generate(iter)
                        if load_gen_img_from_file:
                            img = PIL.Image.open(f"gen/{to_.name}/{to_.dataset}/{str(iter).zfill(6)}.jpg").convert('RGB')
                            img = transforms.ToTensor()(img).unsqueeze(0)
                        from_latent = from_.encode(img)
                else:
                    if from_.dataset == "cifar":
                        img = data[iter][0].reshape(1, 3, 32, 32)
                    else:
                        raise NotImplementedError
                
                if save_pickles:
                    torch.save(from_latent, f"latents/{from_.name}/{from_.dataset}/{iter}_z.pth")
                    torch.save(to_latent, f"latents/{to_.name}/{to_.dataset}/{iter}_z.pth")
                
            pred_to_latent = map(from_latent.flatten())

            loss = criterion(pred_to_latent, to_latent.flatten())

            loss.backward()
            epoch_loss += loss.item()

            if iter % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        losses.append(epoch_loss/n_datapoints)
        print(f"Epoch {epoch} Loss: {epoch_loss/n_datapoints}")

    if plot_losses:
        plt.scatter(range(n_epochs), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"[{from_.dataset}] {from_.name} {from_.latent_layer} to {to_.name} {to_.latent_layer} mapping")
        plt.savefig(f"plots/{from_.dataset}_{from_.name}_{to_.name}.png")
        plt.show()

    torch.save(map, f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}.pth")

def test_latent_space_mapping(from_, to_, start, end, gen_image=False, classnames=[""], load_gen_img_from_file=False):
    map = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}.pth")

    latent_mses = []
    pixel_mses = []

    from_latent = to_latent = None
    for class_ in classnames:
        print(class_)
        if (start == None or end == None) and class_ == "":
            # generate 10 random seeds
            seeds = np.random.randint(0, 10000, 10)
        elif (start == None or end == None) and class_ != "":
            # get names of images in class folder
            seeds = [int(filename.split(".")[0]) for filename in os.listdir(f"data/{to_.dataset}/{class_}") if filename.endswith(".jpg")]
        else:
            seeds = range(start, end)

        for seed in seeds:
            if from_.name == "gan":
                assert gen_image
                from_latent, img = from_.generate(seed)
                to_latent = to_.encode(img)
            elif to_.name == "gan" and gen_image:
                to_latent, img = to_.generate(seed)
                # load image from saved for standardization
                if load_gen_img_from_file:
                    img = PIL.Image.open(f"gen/{to_.name}/{to_.dataset}/{str(seed).zfill(6)}.jpg").convert('RGB')
                    img = transforms.ToTensor()(img).unsqueeze(0)
                from_latent = from_.encode(img)
            else:
                img = PIL.Image.open(f"data/{to_.dataset}{'/' + class_ if class_ != '' else ''}/{str(seed).zfill(6)}.jpg").convert('RGB')
                img = transforms.CenterCrop(from_.pixel_shape[2])(img)
                img = transforms.Resize(size=from_.pixel_shape[2], antialias=True)(img)
                img = transforms.ToTensor()(img).unsqueeze(0)

                from_latent = from_.encode(img)
                if to_.name != "gan":
                    to_latent = to_.encode(img)

            if to_.latent_layer == "w+" and to_latent is not None:
                to_latent = to_.decode_partial(to_latent)

            pred_to_latent = map(from_latent.flatten())

            if to_latent is not None:
                mse = nn.MSELoss()(pred_to_latent, to_latent.flatten())
                print(f">> [{seed}] MSE (latent space): {mse}")
                latent_mses.append(mse.item())
                        
            from_decoded = from_.decode(from_latent)
            to_decoded_pred = to_.decode(pred_to_latent)

            mse = nn.MSELoss()(to_decoded_pred, img)
            print(f">> [{seed}] MSE (pixel space): {mse}")
            pixel_mses.append(mse.item())

            from_.save_image(img, f"mapped/{to_.dataset}/{class_}{seed}{'_gen' if gen_image else ''}_orig.jpg")
            from_.save_image(from_decoded, f"mapped/{to_.dataset}/{class_}{seed}{'_gen' if gen_image else ''}_{from_.name}.jpg")
            to_.save_image(to_decoded_pred, f"mapped/{to_.dataset}/{class_}{seed}{'_gen' if gen_image else ''}_{from_.name}_to_{to_.name}_{to_.latent_layer}.jpg")
            
            if to_latent is not None:
                to_decoded = to_.decode(to_latent)
                print(f">> [{seed}] MSE (direct reconstruction): {nn.MSELoss()(to_decoded_pred, to_decoded)}")
                to_.save_image(to_decoded, f"mapped/{to_.dataset}/{class_}{seed}{'_gen' if gen_image else ''}_{to_.name}.jpg")
    
    # return the indices of the 5 images with the lowest latent space MSEs and lowest pixel space MSEs
    latent_mses = np.array(latent_mses)
    pixel_mses = np.array(pixel_mses)

    print(f">> Average latent space MSE: {np.mean(latent_mses)}")
    print(f">> Average pixel space MSE: {np.mean(pixel_mses)}")

    latent_mses = np.argsort(latent_mses)
    pixel_mses = np.argsort(pixel_mses)


    print(f">> Lowest latent space MSEs: {latent_mses[:5] + start}")
    print(f">> Lowest pixel space MSEs: {pixel_mses[:5] + start}")

    # return the indices of the 5 images with the highest latent space MSEs and highest pixel space MSEs
    print(f">> Highest latent space MSEs: {latent_mses[-5:] + start}")
    print(f">> Highest pixel space MSEs: {pixel_mses[-5:] + start}")

# TODO: refactor this function
def test_double_latent_space_mapping(from_, to_, data, start, end):
    for seed in range(start, end):
        map1 = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}.pth")
        map2 = torch.load(f"pickles/latent_mapping_{to_.name}_{from_.name}_{from_.dataset}.pth")

        if from_.name == "gan":
            from_latent, img = from_.generate(seed)
        elif to_.name == "gan":
            _, img = to_.generate(seed)
            from_latent = from_.encode(img)
        else:
            assert data is not None
            img = data[seed][0].reshape(1, 3, 32, 32)
            from_latent = from_.encode(img)

        pred_to_latent = map1(from_latent.flatten())
        pred_from_latent = map2(pred_to_latent)

        from_decoded_pred = from_.decode(pred_from_latent.view(from_.latent_shape))

        from_.save_image(from_decoded_pred, f"mapped/{to_.dataset}/{seed}_{from_.name}_to_{to_.name}_to_{from_.name}.jpg")

def load_cifar():
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    transform = transforms.Compose([transforms.ToTensor()])
    data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    return data
    
if __name__ == "__main__":

    # gan = Model("gan", "pickles/stylegan-cifar.pkl", "cifar", "w+")
    gan = Model("gan", "pickles/stylegan-celeba.pkl", "celeba", "z")
    vae = Model("vae", "pickles/vae-celeba.pth", "celeba")



    attributes = pd.read_csv("list_attr_celeba.csv")
    # get first 10 images with positive values for each attribute and move into appropriate folder
    # for column in attributes.columns[1:]:
    #     # make new directory for attribute wiping anything that was previously there
    #     if os.path.exists(f"data/celeba/{column}"):
    #         os.system(f"rm -rf data/celeba/{column}")
        
    #     os.mkdir(f"data/celeba/{column}")
    #     for index, row in attributes[attributes[column] == 1].head(10).iterrows():
    #         # copy rather than move the image
    #         os.system(f"cp data/celeba/{str(row['image_id']).zfill(6)} data/celeba/{column}/{str(row['image_id']).zfill(6)}")

    # nf = Model("nf", "pickles/nf-cifar.pt", "cifar")
    print(attributes.columns[1:])
    train_args = {
        "from_": vae,
        "to_": gan,
        "data": None,
        "n_epochs": 10,
        "n_datapoints": 2000,
        "criterion": nn.MSELoss(),
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "momentum": None,
        "batch_size": 100,
        "from_saved": True,
        "save_pickles": True,
        "plot_losses": True,
        "load_gen_img_from_file": True
    }

    test_args = {
        "from_": vae,
        "to_": gan,
        "start": None,
        "end": None,
        "gen_image": False,
        "classnames": attributes.columns[1:].tolist(),
        "load_gen_img_from_file": True
    }

    # train_latent_space_mapping(**train_args)
    test_latent_space_mapping(**test_args)
    # test_latent_space_mapping(nf, gan, None, 10000, 10010, True)

    # test_latent_space_mapping(vae, gan, "", 0, 10, False)
        

        
