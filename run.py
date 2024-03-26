import os
import sys
import json
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from joblib import dump, load
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from glow.datasets import postprocess, preprocess
from glow.model import Glow
from glow_pytorch.model import Glow as Glow_c
from glow_pytorch.train import calc_z_shapes
from diffusers import VQModel

from cifar_ae import create_model
import legacy

device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")

def log(msg):
    print(msg)
    LOG.write(msg + "\n")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Model():
    def __init__(self, model, name, pickle, dataset, latent_layer="z"):
        self.model = model
        self.name = name
        self.dataset = dataset
        self.model_ = None
        self.pixel_shape = None
        self.z_shape = None
        self.z_dim = None
        self.latent_shape = None
        self.latent_dim = None
        self.latent_layer = latent_layer
        
        # temporarily redirect stdout to devnull to suppress printing of state dict keys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')     

        if self.model == "nf" and self.dataset == "cifar": # normalizing flow
            with open("glow/hparams.json") as json_file:  
                hparams = json.load(json_file)
                
            self.model_ = Glow((32, 32, 3), hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                        hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], 10,
                        hparams['learn_top'], hparams['y_condition'])

            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
            self.model_.set_actnorm_init()
            self.model_ = self.model_.eval()
            sys.stdout = original_stdout

            self.z_shape = self.latent_shape = (1, 48, 4, 4)
            self.z_dim = self.latent_dim = 48 * 4 * 4
        elif self.model == "nf" and self.dataset == "celeba":
            self.model_ = Glow_c(3, 32, 4, affine=False, conv_lu=True)
            self.model_ = nn.DataParallel(self.model_)
            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
            self.model_ = self.model_.module
            self.model_.eval()

            self.z_shape = [(0, 0, 0)]
            self.z_shape += calc_z_shapes(3, 64, 32, 4)
            self.latent_shape = self.z_shape
            # self.z_shape = (3, 64, 64)
            self.z_dim = self.latent_dim = 3 * 64 * 64
            sys.stdout = original_stdout
        elif self.model == "ae":
            self.model_ = create_model()
            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
            sys.stdout = original_stdout

            assert self.dataset == "cifar"
            self.z_shape = self.latent_shape = (1, 48, 4, 4)
            self.z_dim = self.latent_dim = 48 * 4 * 4
        elif self.model == "vae":
            self.model_ = torch.load(pickle, map_location='cpu')
            sys.stdout = original_stdout
            assert self.dataset == "celeba"

            self.z_shape = self.latent_shape = (1, 128)
            self.z_dim = self.latent_dim = 128
        elif self.model == "vqvae":
            self.model_ = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
            sys.stdout = original_stdout
            assert self.dataset == "celeba"

            self.z_shape = self.latent_shape = (1, 3, 37, 37) # (1, 3, 64, 64)
            self.z_dim = self.latent_dim = 1 * 3 * 37 * 37
        elif self.model == "gan":
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
        elif self.dataset == "celeba" and self.model == "nf":
            self.pixel_shape = (1, 3, 64, 64)            
        elif self.dataset == "celeba" or self.dataset == "metfaces":
            self.pixel_shape = (1, 3, 150, 150)
            # self.pixel_shape = (1, 3, 256, 256)

        if self.model_ is None: raise NotImplementedError
        
        print(f">> ###### Loaded {self.model} model ###### ")
        print(f">> Dataset: {self.dataset}")
        print(f">> Pixel shape: {self.pixel_shape}")
        print(f">> z shape: {self.z_shape}")
        print(f">> Latent layer: {self.latent_layer}")
        print(f">> Latent shape: {self.latent_shape}")

    def encode(self, img):
        if self.model == "nf" and self.dataset == "cifar":
            z, _, _ = self.model_(preprocess(img))
            return z
        elif self.model == "nf" and self.dataset == "celeba":
            img = transforms.Resize(self.pixel_shape[2], antialias=True)(img)
            img = transforms.CenterCrop(self.pixel_shape[2])(img)
            _, _, z_ = self.model_(img)

            z = torch.concat([torch.flatten(elem) for elem in z_], dim=0)
            return z
        elif self.model == "ae":
            return self.model_.encoder(img)
        elif self.model == "vae":
            mu, log_var = self.model_.encode(img)
            return self.model_.reparameterize(mu, log_var)
        elif self.model == "vqvae":
            return self.model_.encode(img).latents
        else:
            raise NotImplementedError(f">> Encode not implemented for {self.model}")
    
    def decode_partial(self, z):
        if self.latent_layer == "z":
            return z
        elif self.latent_layer == "w+":
            return self.model_.mapping(z, None)[:, 0, :]

        raise NotImplementedError

    def decode(self, latent):
        """
        Decodes the given latent vector into an image.

        Args:
            latent (torch.Tensor): The latent vector to decode into an image.
        Returns:
            torch.Tensor: The decoded image in pixel space.
        """
        if not (self.model == "nf" and self.dataset == "celeba"):
            latent = latent.view(self.z_shape)
        

        if self.model == "nf" and self.dataset == "cifar":
            # latent = torch.clamp(latent, min=0)
            return self.model_(y_onehot=None, z=latent, temperature=0, reverse=True).cpu()
        elif self.model == "nf" and self.dataset == "celeba":
            latents = []
            z_prev_size = 0
            z_curr_size = 0
            for i in range(1, len(self.z_shape)):
                z_prev_size += self.z_shape[i - 1][0] * self.z_shape[i - 1][1] * self.z_shape[i - 1][2]
                z_curr_size += self.z_shape[i][0] * self.z_shape[i][1] * self.z_shape[i][2]
                latents.append(latent[z_prev_size:z_curr_size].view(1, *self.z_shape[i]))

            return self.model_.reverse(latents, reconstruct=True).cpu().data
        elif self.model == "ae":
            return self.model_.decoder(latent)
        elif self.model == "vae":
            return self.model_.decode(latent).view(self.pixel_shape)
        elif self.model == "vqvae":
            return self.model_.decode(latent).sample
        elif self.model == "gan":
            if self.latent_layer == "z":
                img = self.model_(latent, None, noise_mode='none', force_fp32=True)
            elif self.latent_layer == "w+":
                img = self.model_.synthesis(latent.repeat([1, self.latent_shape[1], 1]), noise_mode='none', force_fp32=True)
            
            if self.pixel_shape[2] != img.shape[2]:
                img = transforms.Resize(size=self.pixel_shape[2], antialias=True)(img)

            return img
        else:
            raise NotImplementedError
    
    def reconstruct(self, img, filename, n_recon=5) :
        if self.model == "nf" and self.dataset == "cifar":
            z = self.encode(img)
            print(z.shape)
            pic = postprocess(self.decode(z)).squeeze()
            plt.imshow(pic.permute(1,2,0))
            plt.savefig(f"rec/{filename}")
        elif self.model == "nf" and self.dataset == "celeba":
            _, _, z = self.model_(img)
            recon = self.model_.reverse(z, reconstruct=True)
            save_image(recon, filename, normalize=True)
        elif self.model == "ae":        
            pics = img
            for _ in range(n_recon):
                pic = self.model_(img)[1]
                pics = torch.cat((pics, recon), dim=0)
                save_image(pic, f"rec/{filename}")
        elif self.model == "vae":
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
        if self.model == "nf" or self.model == "ae" or self.model == "vae" or self.model == "vqvae":
            save_image(img, filename, normalize=True)
        else:
            img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(filename)
    
    def generate(self, seed):
        if self.model == "nf" and self.dataset == "cifar":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            img = postprocess(self.model_(y_onehot=None, z=latent, temperature=1, reverse=True))
        elif self.model == "nf" and self.dataset == "celeba":
            z_sample = []
            z_new = torch.from_numpy(np.random.RandomState(seed).randn(*self.z_shape)).to(torch.float).to(device)
            z_prev_size = 0
            z_curr_size = 0
            for i in range(1, len(self.latent_shape)):
                z_prev_size += self.latent_shape[i - 1][0] * self.latent_shape[i - 1][1] * self.latent_shape[i - 1][2]
                z_curr_size += self.latent_shape[i][0] * self.latent_shape[i][1] * self.latent_shape[i][2]
                z_new_ = z_new[z_prev_size:z_curr_size].view(1, *self.latent_shape[i]) * 0.7
                z_sample.append(z_new_.to(device))

            img = self.model_.reverse(z_sample).cpu().data
        elif self.model == "ae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            img = self.model_.decoder(latent)
        elif self.model == "vae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            img = self.model_.decode(latent).view(self.pixel_shape)
        elif self.model == "vqvae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape)).to(torch.float).to(device)
            z = latent
            img = self.model_.decode(latent).sample
        elif self.model == "gan":
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.model_.z_dim)).to(torch.float32).to(device)
            # TODO: uncomment these
            latent = self.model_.mapping(z, None)
            # img = self.model_.synthesis(latent * 0.8 + 0.2, noise_mode='none', force_fp32=True)
            img = self.model_(z, None, noise_mode='none', force_fp32=True)

            if self.pixel_shape[2] != img.shape[2]:
                img = transforms.Resize(size=self.pixel_shape[2], antialias=True)(img)
            
            if self.latent_layer == "w+":
                z = latent[:, 0, :]
        
        self.save_image(img, f'gen/{self.model}/{self.dataset}/{str(seed).zfill(6)}.jpg')

        return z, img

class OneLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def map_norm(from_, to_, latent_layer="z", regularized=False):
    map = load(f"pickles/latent_mapping_{from_}_{to_}_celeba_{latent_layer}_linear{'_unreg' if not regularized else ''}.joblib")
    print(np.linalg.norm(map.coef_))

def train_exact_latent_space_mapping(
        from_name, 
        to_name,
        dataset, 
        n_train,
        synthetic=False,
        regularized=False,
    ):
    real = '' if synthetic else '-real'

    from_directory = f"latents/{from_name}/{dataset}{real}"
    to_directory = f"latents/{to_name}/{dataset}{real}"

    from_latents_train = [torch.load(f"{from_directory}/{i if synthetic or dataset == 'cifar' else str(i).zfill(6)}_{'w+' if from_name == 'gan' else 'z'}.pth").flatten().detach() for i in range(1, n_train)] # 'w+' if from_name == 'gan' else 'z'
    to_latents_train = [torch.load(f"{to_directory}/{i if synthetic or dataset == 'cifar' else str(i).zfill(6)}_{'w+' if to_name == 'gan' else 'z'}.pth").flatten().detach() for i in range(1, n_train)]
    from_latents_train = torch.stack(from_latents_train, 0).numpy()
    to_latents_train = torch.stack(to_latents_train, 0).numpy()

    print(from_latents_train[0].shape)
    print(to_latents_train[0].shape)

    if regularized and to_name == 'gan':
        reg = Lasso(alpha=0.001, max_iter=10).fit(from_latents_train, to_latents_train)
    elif regularized:
        reg = Ridge(alpha=50000).fit(from_latents_train, to_latents_train)
    else:
        print("HERE")
        reg = LinearRegression().fit(from_latents_train, to_latents_train)

    print(reg.score(from_latents_train, to_latents_train))

    dump(reg, f"pickles/latent_mapping_{from_name}_{to_name}_{dataset}_{'w+' if to_name == 'gan' else 'z'}_linear{'_unreg' if not regularized else ''}.joblib") # 'w+' if to_name == 'gan' else 'z'

def test_exact_latent_space_mapping(
        from_,
        to_,
        test_range=None,
        synthetic_test=False
    ):

    real_test = '' if synthetic_test else '-real'
    from_directory_test = f"latents/{from_.name}/{to_.dataset}{real_test}"
    to_directory_test = f"latents/{to_.name}/{to_.dataset}{real_test}"

    if test_range is None:
        test_range = list(range(1950, 1955)) if to_.dataset == 'cifar' else list(range(14950, 15000))

    from_latents_test = [torch.load(f"{from_directory_test}/{i if synthetic_test or to_.dataset == 'cifar' else str(i).zfill(6)}_{from_.latent_layer}.pth").flatten().detach() for i in test_range]
    from_latents_test = torch.stack(from_latents_test, 0).numpy()

    
    reg = load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}_linear.joblib")
    predictions = reg.predict(from_latents_test)

    test_mse = None
    if (synthetic_test and to_.name == 'gan') or (not synthetic_test and to_.name != 'gan'):
        to_latents_test = [torch.load(f"{to_directory_test}/{i if synthetic_test or to_.dataset == 'cifar'  else str(i).zfill(6)}_{to_.latent_layer}.pth").flatten().detach() for i in test_range]
        to_latents_test = torch.stack(to_latents_test, 0).numpy()
        test_mse = np.mean((to_latents_test - predictions) ** 2)
        log(f"Test MSE: {np.mean((to_latents_test - predictions) ** 2)}")

    for seed in range(len(predictions)):
        from_.save_image(from_.decode(torch.Tensor(from_latents_test[seed])), f"mapped/{to_.dataset}/{str(test_range[0] + seed).zfill(6)}_{'gen_' if synthetic_test else ''}{from_.name}.jpg")
        if (synthetic_test and to_.name == 'gan') or (not synthetic_test and to_.name != 'gan'):
            to_.save_image(to_.decode(torch.Tensor(to_latents_test[seed])), f"mapped/{to_.dataset}/{str(test_range[0] + seed).zfill(6)}_{'gen_' if synthetic_test else ''}{to_.name}_w.jpg")

        # print(predictions[seed])
        to_.save_image(to_.decode(torch.Tensor(predictions[seed])), f"mapped/{to_.dataset}/{str(test_range[0] + seed).zfill(6)}_{'gen_' if synthetic_test else ''}{from_.name}_to_{to_.name}_{to_.latent_layer}_linear.jpg")

    return test_mse

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
        load_gen_img_from_file=False,
        map_type="linear"
    ):
    print(f">> Training latent space mapping from {from_.name} to {to_.name} over {n_datapoints} datapoints and {n_epochs} epochs")
    
    if map_type == "linear":
        map = nn.Linear(from_.z_dim, to_.z_dim)
        # map = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}_{map_type}.pth")
    elif map_type == "nonlinear-1":
        map = OneLayerNN(from_.z_dim, to_.z_dim)
    else:
        raise(NotImplementedError(f"map type {map_type} not implemented"))

    if optimizer == "adam":
        optimizer = optim.Adam(map.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(map.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise NotImplementedError

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for iter in range(1, n_datapoints): # added the 1 for celeba
            # print(iter)
            if save_pickles and (from_saved or epoch > 0):
                from_latent = torch.load(f"latents/{from_.name}/{from_.dataset}/{iter}_{from_.latent_layer}.pth")
                to_latent = torch.load(f"latents/{to_.name}/{to_.dataset}/{iter}_{to_.latent_layer}.pth") # {to_.latent_layer} {'_z' if to_.latent_layer == 'metfaces' else ''}
            else:
                # use GAN to generate the data
                if data == None:
                    if from_.model == "gan":
                        # from_latent, img = from_.generate(iter)
                        from_latent = torch.load(f"latents/{from_.name}/{to_.dataset}/{iter}_w+.pth")
                        if load_gen_img_from_file:
                            img = PIL.Image.open(f"gen/{from_.name}/{to_.dataset}/{str(iter).zfill(6)}.jpg").convert('RGB')
                            img = transforms.ToTensor()(img).unsqueeze(0)
                        to_latent = to_.encode(img)
                    elif to_.model == "gan":
                        # to_latent, img = to_.generate(iter)
                        to_latent = torch.load(f"latents/{to_.name}/{to_.dataset}/{'_z' if to_.dataset != 'metfaces' else ''}.pth")
                        if load_gen_img_from_file:
                            img = PIL.Image.open(f"gen/{to_.name}/{to_.dataset}/{str(iter).zfill(6)}.jpg").convert('RGB')
                            img = transforms.ToTensor()(img).unsqueeze(0)
                        from_latent = from_.encode(img)
                else:
                    img = PIL.Image.open(f"{data}/{str(iter + 1).zfill(6)}.jpg").convert('RGB')
                    img = transforms.CenterCrop(from_.pixel_shape[2])(img)
                    img = transforms.Resize(size=from_.pixel_shape[2], antialias=True)(img)
                    img = transforms.ToTensor()(img).unsqueeze(0)

                    to_latent = to_.encode(img)
                    from_latent = from_.encode(img)
                
                if save_pickles:
                    torch.save(from_latent, f"latents/{from_.name}/{to_.dataset}/{iter}_z.pth")
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
        plt.title(f"[{from_.dataset}] {from_.name} {from_.latent_layer} to {to_.name} {to_.latent_layer} {map_type} mapping")
        plt.savefig(f"plots/{from_.dataset}_{from_.name}_{from_.latent_layer}_{to_.name}_{to_.latent_layer}_{map_type}.png")
        plt.show()
    
    log(f"Train Loss: {losses[-1]}")

    torch.save(map, f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}_{map_type}.pth")

def test_latent_space_mapping(from_, to_, start, end, gen_image=False, classnames=[""], load_gen_img_from_file=False, map_type="linear"):
    map = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_{to_.latent_layer}_{map_type}.pth") #
    print(map)

    latent_mses = []
    pixel_mses = []

    from_latent = to_latent = None
  
    seeds = range(start, end)

    for seed in seeds:
        if from_.model == "gan" and to_.model == "gan":
            assert from_.dataset == to_.dataset
            from_latent, img = from_.generate(seed)
            to_latent, img = to_.generate(seed)
        elif from_.model == "gan" and gen_image:
            from_latent = torch.load(f"latents/{from_.model}/{from_.dataset}/{seed}_w+.pth")
            if load_gen_img_from_file:
                img = PIL.Image.open(f"gen/{from_.model}/{to_.dataset}/{str(seed).zfill(6)}.jpg").convert('RGB')
                img = transforms.ToTensor()(img).unsqueeze(0)
            to_latent = to_.encode(img)
        elif to_.model == "gan" and gen_image:
            if load_gen_img_from_file:
                img = PIL.Image.open(f"gen/{to_.model}/{to_.dataset}/{str(seed).zfill(6)}.jpg").convert('RGB')
                img = transforms.ToTensor()(img).unsqueeze(0)
            from_latent = from_.encode(img)
        elif gen_image:
            from_latent = torch.load(f"latents/{from_.name}/{from_.dataset}/{seed}_z.pth")
            to_latent = torch.load(f"latents/{to_.name}/{to_.dataset}/{seed}_z.pth")
        else:
            from_latent = torch.load(f"latents/{from_.name}/{from_.dataset}-real/{str(seed).zfill(6)}_z.pth")

        pred_to_latent = map(from_latent.flatten())

        if to_latent is not None:
            mse = nn.MSELoss()(pred_to_latent, to_latent.flatten())
            log(f">> [{seed}] MSE (latent space): {mse}")
            latent_mses.append(mse.item())
        
        from_decoded = from_.decode(from_latent)
        to_decoded_pred = to_.decode(pred_to_latent)

        from_.save_image(from_decoded, f"mapped/{to_.dataset}/{seed}{'_gen' if gen_image else ''}_{from_.name}.jpg")
        to_.save_image(to_decoded_pred, f"mapped/{to_.dataset}/{seed}{'_gen' if gen_image else ''}_{from_.name}_to_{to_.name}_{to_.latent_layer}_{map_type}.jpg")
        
        if to_latent is not None:
            to_decoded = to_.decode(to_latent)
            to_.save_image(to_decoded, f"mapped/{to_.dataset}/{seed}{'_gen' if gen_image else ''}_{to_.name}.jpg")
    
    # return the indices of the 5 images with the lowest latent space MSEs and lowest pixel space MSEs
    latent_mses = np.array(latent_mses)
    pixel_mses = np.array(pixel_mses)

    print(f">> Average latent space MSE: {np.mean(latent_mses)}")
    print(f">> Average pixel space MSE: {np.mean(pixel_mses)}")

    latent_mses = np.argsort(latent_mses)
    pixel_mses = np.argsort(pixel_mses)


    print(f">> Lowest latent space MSEs: {latent_mses[:5] + start}")
    print(f">> Lowest pixel space MSEs: {pixel_mses[:5] + start}")

    print(f">> Highest latent space MSEs: {latent_mses[-5:] + start}")
    print(f">> Highest pixel space MSEs: {pixel_mses[-5:] + start}")

if __name__ == "__main__":
    # CIFAR-10 Models
    # nf = Model("nf", "nf", "pickles/nf-cifar.pt", "cifar", "z")
    # ae = Model("ae", "ae1", "pickles/ae1-cifar.pkl", "cifar", "z")
    # gan = Model("gan", "gan", "pickles/stylegan-cifar.pkl", "cifar", "w+")

    # CelebA Models
    # vqvae = Model("vqvae", "vqvae", None, "celeba", "z")
    # vae = Model("vae", "vae-21", "pickles/vae_model_21.pth", "celeba", "z")
    # nf = Model("nf", "nf-celeba", "pickles/nf-celeba.pt", "celeba", "z")
    # gan = Model("gan", "gan", "pickles/stylegan-celeba.pkl", "celeba", "w+")

    #! Plotting latent space MSE
    # LOG = open(f"celeba_logs.txt", "w")
    # models = [gan, vae, vqvae, nf]
    # results = []

    # for i in range(len(models)):
    #     for j in range(len(models)):
    #         if models[i].name != 'gan' and models[j].name != 'gan' and i != j:
    #             continue
    #             log(f"Testing {models[i].name} -> {models[j].name} mapping")
    #             latent_mse = test_exact_latent_space_mapping(models[i], models[j], synthetic_test=False)
            
    #             results.append(
    #                 {
    #                     "Encoder": models[i].model,
    #                     "Decoder": models[j].model,
    #                     "Latent MSE": latent_mse
    #                 }
    #             )
    #         elif models[j].name == 'gan' and i != j:
    #             log(f"Testing {models[i].name} -> {models[j].name} mapping")
    #             latent_mse = test_exact_latent_space_mapping(models[i], models[j], synthetic_test=True)
            
    #             results.append(
    #                 {
    #                     "Encoder": models[i].model,
    #                     "Decoder": models[j].model,
    #                     "Latent MSE": latent_mse
    #                 }
    #             )
                
    #         elif models[i].name != 'gan' and i == j:
    #             results.append(
    #                 {
    #                     "Encoder": models[i].model,
    #                     "Decoder": models[j].model,
    #                     "Latent MSE": 0
    #                 }
    #             )


    # LOG.close()

    # df = pd.DataFrame(results)
    # print(df)
    # plt.figure(figsize=(4, 3))
    # df.Encoder=pd.Categorical(df.Encoder,categories=df.Encoder.unique(),ordered=True)
    # df.Decoder=pd.Categorical(df.Decoder,categories=df.Decoder.unique(),ordered=True)
    # pivoted = df.pivot("Encoder", "Decoder", "Latent MSE")
    # # pivoted.sort_index(level=1, ascending=True, inplace=True)

    # sns.heatmap(pivoted, annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True))
    # plt.title("Latent MSE Heatmap")
    # plt.tight_layout()
    # plt.savefig(f"plots/latent_mse_cifar.png")

    #! Reconstructing Images
    # for i in range(1, 6):
    #     img = PIL.Image.open(f"data/celeba/00000{i}.jpg")
    #     img = transforms.CenterCrop(160)(img)
    #     img = transforms.Resize(size=64, antialias=True)(img)
    #     img = transforms.ToTensor()(img).unsqueeze(0)
    #     nf.reconstruct(img, f"test_{i}.jpg")