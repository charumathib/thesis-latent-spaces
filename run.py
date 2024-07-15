import json
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from joblib import dump, load
from sklearn.linear_model import LinearRegression, Ridge
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
from ddim.runners.diffusion import Diffusion
from ddim.models.diffusion import Model as DiffusionModel
from ddim.main import parse_args_and_config
from ddim.models.ema import EMAHelper
from ddim.datasets import inverse_data_transform

from DiffuseVAE.main.models.vae import VAE

from glow.datasets import postprocess, preprocess
from glow.model import Glow
from glow_pytorch.model import Glow as Glow_c
from glow_pytorch.train import calc_z_shapes
from diffusers import VQModel, UNet2DModel, DPMSolverMultistepScheduler, DDIMInverseScheduler
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.icml2024())
import legacy

device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Model():
    def __init__(self, model, dataset, name=None, use_thresholding=False):
        # note: use_thresholding is only used for DPM
        self.model = model
        self.name = name if name is not None else self.model
        self.dataset = dataset
        self.pixel_shape = (3, 64, 64) if dataset == "celeba" else (3, 32, 32)
        self.model_ = None
        self.z_shape = None
        self.z_dim = None
        self.latent_shape = None
        self.latent_dim = None
        
        if self.model == "dm":
            if self.dataset == "celeba":
                self.z_shape = self.latent_shape = (1, 3, 64, 64)
                self.z_dim = self.latent_dim = 3 * 64 * 64

                args, self.config = parse_args_and_config()
                self.runner_ = Diffusion(args, self.config)
                self.model_ = DiffusionModel(self.config)

                states = torch.load("pickles/dm-celeba.pth", map_location='cpu')

                self.model_ = self.model_.to("cpu")
                self.model_ = torch.nn.DataParallel(self.model_)
                self.model_.load_state_dict(states[0], strict=True)

                ema_helper = EMAHelper(mu=self.config.model.ema_rate) 
                ema_helper.register(self.model_)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(self.model_)

                self.model_.eval()
            elif self.dataset == "cifar":
                model_id = "google/ddpm-cifar10-32"
                self.model_ = UNet2DModel.from_pretrained(model_id)
                # self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id)
                self.scheduler = DPMSolverMultistepScheduler(thresholding=use_thresholding,  solver_order=3, dynamic_thresholding_ratio=0.995, sample_max_value=2.0)
                # self.scheduler.thresholding = True
                # print(self.scheduler.__dict__)
                # self.invscheduler = DPMSolverMultistepInverseScheduler.from_pretrained(model_id)
                # self.invscheduler = DPMSolverMultistepInverseScheduler.from_pretrained(model_id)
                self.invscheduler = DDIMInverseScheduler.from_pretrained(model_id)
                self.z_shape = (1, 3, 32, 32)
                self.z_dim = 3072
                self.latent_shape = self.z_shape
                self.latent_dim = self.z_dim
        elif self.model == "vae-diffusion":
            self.z_shape = (1, 512, 1, 1)
            self.z_dim = 512
            self.latent_shape = self.z_shape
            self.latent_dim = self.z_dim
            if self.dataset == "cifar":
                vae_ckpt = "pickles/vae-cifar.ckpt"
                
            elif self.dataset == "celeba":
                vae_ckpt = "pickles/vae-celeba.ckpt"

            self.vae_model_ = VAE.load_from_checkpoint(vae_ckpt, input_res=self.pixel_shape[1])
            self.vae_model_.eval()
        elif self.model == "nf":
            if self.dataset == "cifar":
                with open("glow/hparams.json") as json_file:  
                    hparams = json.load(json_file)
                    
                self.model_ = Glow((32, 32, 3), hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                            hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], 10,
                            hparams['learn_top'], hparams['y_condition'])

                self.model_.load_state_dict(torch.load("pickles/nf-cifar.pt", map_location='cpu'))
                self.model_.set_actnorm_init()
                self.model_ = self.model_.eval()

                self.z_shape = self.latent_shape = (1, 48, 4, 4)
                self.z_dim = self.latent_dim = 48 * 4 * 4
            elif self.dataset == "celeba":
                self.model_ = Glow_c(3, 32, 4, affine=False, conv_lu=True)
                self.model_ = nn.DataParallel(self.model_)
                self.model_.load_state_dict(torch.load("pickles/nf-celeba.pt", map_location='cpu'))
                self.model_ = self.model_.module
                self.model_.eval()

                self.z_shape = [(0, 0, 0)]
                self.z_shape += calc_z_shapes(3, 64, 32, 4)
                self.latent_shape = self.z_shape
                self.z_dim = self.latent_dim = 3 * 64 * 64
        elif self.model == "vqvae":
            self.model_ = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
            assert self.dataset == "celeba"

            self.z_shape = self.latent_shape = (1, 3, 16, 16) # (1, 3, 64, 64)
            self.z_dim = self.latent_dim = 1 * 3 * 16 * 16
        elif self.model == "gan":
            if self.dataset == "celeba":
                with open("pickles/gan-celeba.pkl", 'rb') as f:
                    self.model_ = pickle.load(f)['G_ema'].cpu().float()

                self.z_shape = (1, 512)
                self.z_dim = 512
                self.latent_shape = (1, 10, 512)
                self.latent_dim = 10 * 512

            elif self.dataset == "cifar":
                with open("pickles/stylegan-cifar.pkl", 'rb') as f:
                    self.model_ = legacy.load_network_pkl(f)['G_ema'].to(device)
                assert self.dataset == "cifar" or self.dataset == "celeba" or self.dataset == "metfaces"

                self.z_shape = self.latent_shape = (1, 512)
                self.z_dim = self.latent_dim = 512

                self.latent_shape = (1, 8, 512)
                self.latent_dim = 8 * 512
        
        print(f">> ###### Loaded {self.model} model ###### ")
        print(f">> Dataset: {self.dataset}")
        print(f">> Pixel shape: {self.pixel_shape}")
        print(f">> Latent shape: {self.z_shape}")

    def encode(self, img):
        if self.model == "nf" and self.dataset == "cifar":
            z, _, _ = self.model_(preprocess(img))
            return z
        elif self.model == "nf" and self.dataset == "celeba":
            _, _, z_ = self.model_(img)
            z = torch.concat([torch.flatten(elem) for elem in z_], dim=0)
            return z
        elif self.model == "dm" and self.dataset == "cifar":
            input = img
            self.invscheduler.set_timesteps(50)
            for t in tqdm(self.invscheduler.timesteps):
                with torch.no_grad():
                    noisy_residual = self.model_(input, t).sample
                previous_noisy_sample = self.invscheduler.step(noisy_residual, t, input).prev_sample
                input = previous_noisy_sample
            
            return input
        elif self.model == "vae-diffusion":
            mu, logvar = self.vae_model_.encode(img * 0.5 + 0.5)
            z = self.vae_model_.reparameterize(mu, logvar).unsqueeze(0)
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
    
    def decode(self, latent):
        if not (self.model == "nf" and self.dataset == "celeba"):
            latent = latent.reshape(self.z_shape)
        
        if self.model == "dm":
            if self.dataset == "celeba":
                img = self.runner_.sample_image(latent, self.model_)
                return img
            elif self.dataset == "cifar":
                input = torch.tensor(latent)
                self.scheduler.set_timesteps(20)
                for t in tqdm(self.scheduler.timesteps):
                    with torch.no_grad():
                        noisy_residual = self.model_(input, t).sample
                    previous_noisy_sample = self.scheduler.step(noisy_residual, t, input).prev_sample
                    input = previous_noisy_sample

                return input
            
        if self.model == "vae-diffusion":
            recons = self.vae_model_.decode(latent)
            # latent = latent.view(1, 3, 64, 64)
            # diff_recon, _ = self.model_.predict_step(recons, 0, recons, None)
            # return next(iter(diff_recon.values()))
            return recons
        
        elif self.model == "nf" and self.dataset == "cifar":
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
            img = self.model_.synthesis(latent.repeat([1, self.latent_shape[1], 1]), noise_mode='none', force_fp32=True)

            return img
        else:
            raise NotImplementedError
    
    def reconstruct(self, img, filename, n_recon=5) :
        if self.model == "nf" and self.dataset == "cifar":
            z = self.encode(img)
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
        
    def save_image(self, img, filename, norm=True):
        if self.model == "nf" or self.model == "ae" or self.model == "vae" or self.model == "vqvae":
            save_image(img, filename, normalize=True)
        else:
            if self.model == "dm" and self.dataset == "cifar":
                img_ = (img.squeeze().permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
                cv2.imwrite(filename, img_)
            elif self.model == "dm" and self.dataset == "celeba":
                img = inverse_data_transform(self.config, img)
                save_image(img, filename, normalize=True)
            else:
                img_ = ((img.squeeze().permute(1, 2, 0) * 127.5 + 128) if norm else (img.squeeze().permute(1, 2, 0) * 255)).round().clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img_.cpu().numpy(), 'RGB').save(filename)
    
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
            latent = self.model_.mapping(z, None)
            img = self.model_(z, None, noise_mode='none', force_fp32=True)

            if self.pixel_shape[2] != img.shape[2]:
                img = transforms.Resize(size=self.pixel_shape[2], antialias=True)(img)
            
            z = latent[:, 0, :]
            
            self.save_image(img, f'data/{self.dataset}_gen/{str(seed).zfill(4 if self.dataset == "cifar" else 6)}.jpg')
        
        if self.model != "gan":
            self.save_image(img, f'gen/{self.model}/{self.dataset}/{str(seed).zfill(6)}.jpg')

        return z, img


def map_norm(from_, to_, latent_layer="z", regularized=False):
    map = load(f"pickles/latent_mapping_{from_}_{to_}_celeba_{latent_layer}_linear{'_unreg' if not regularized else ''}.joblib")
    print(np.linalg.norm(map.coef_))

def train_exact_latent_space_mapping(
        from_name, 
        to_name,
        dataset, 
        synthetic=False,
        regularized=False,
    ):

    from_latents_train, _ = get_latents(from_name, dataset, train=True, synthetic=synthetic)
    to_latents_train, _ = get_latents(to_name, dataset, train=True, synthetic=synthetic)

    print(from_latents_train[0].shape)
    print(to_latents_train[0].shape)

    if regularized:
        if from_name == "dm" and to_name == "vqvae":
            alpha = 5000
        elif from_name == "dm" and to_name == "nf":
            alpha = 5000
        elif from_name == "nf" and to_name == "gan":
            alpha = 50000
        elif from_name == "nf" and to_name == "vqvae":
            alpha = 50000
        elif from_name == "nf" and to_name == "dm":
            alpha = 50000
        elif from_name == "dm" and to_name == "vae-diffusion":
            alpha = 100
        elif from_name == "nf" and to_name == "vae-diffusion":
            alpha = 5000
        elif from_name == "dm" and to_name == "gan":
            alpha = 2000
        elif from_name == "nf" and to_name == "dm" and dataset == "cifar":
            alpha = 1000000000

        reg = Ridge(alpha=alpha).fit(from_latents_train, to_latents_train)
    else:
        reg = LinearRegression().fit(from_latents_train, to_latents_train)

    print(reg.score(from_latents_train, to_latents_train))

    dump(reg, f"pickles/latent_mapping_{from_name}_{to_name}_{dataset}_linear{'_reg' if regularized else ''}.joblib")

def get_latents(model_name, dataset, train, synthetic=False):
        if dataset == "celeba":
            range_ = range(1, 9001) if train else range(9001, 9101)
            latents = [torch.load(f"latents/{model_name}/celeba_{'gen' if synthetic else 'real'}/{str(i).zfill(6)}.pth").flatten().detach() for i in range_]
        elif dataset == "cifar":
            range_ = []
            for i in range(10):
                range_ += list(range((400 * i) + 1, (400 * i) + 401)) if train else list(range((500 * i) + 401, (500 * i) + 411))
            
            latents = [torch.load(f"latents/{model_name}/cifar_{'gen' if synthetic else 'real'}/{str(i).zfill(4)}.pth", map_location='cpu').flatten().detach() for i in range_]

        return torch.stack(latents, 0).numpy(), range_

def test_exact_latent_space_mapping(
        from_name,
        to_name,
        dataset,
        on_train_set=False,
        synthetic=False,
        save_images=False,
        regularized=False,
        ind=None
    ):

    from_latents_test, range_ = get_latents(from_name, dataset, train=on_train_set, synthetic=synthetic)
    
    reg = load(f"pickles/latent_mapping_{from_name}_{to_name}_{dataset}_linear{'_reg' if regularized else ''}.joblib")
    predictions = reg.predict(from_latents_test)

    latent_mse = None
    if synthetic or (from_name != "gan" and to_name != 'gan'):
        to_latents_test, _ = get_latents(to_name, dataset, train=on_train_set, synthetic=synthetic)
        if ind is not None:
            latent_mse = np.mean((to_latents_test[range_.index(ind)] - predictions[range_.index(ind)]) ** 2)
        else:
            latent_mse = np.mean((to_latents_test - predictions) ** 2)

    if save_images:
        if ind is not None:
            range_ = ind

        to_ = Model(to_name, dataset)
        for i, seed in tqdm(enumerate(range_)):
            decoded_mapped = to_.decode(torch.Tensor(predictions[i]))
            to_.save_image(decoded_mapped, f"mapped/{dataset}/{str(seed).zfill(6)}_{'gen_' if synthetic else ''}{from_name}_to_{'vae' if to_name == 'vae-diffusion' else to_name}_linear.jpg", norm=to_.name != 'vae-diffusion')

    return latent_mse

def test_cifar_mappings():
    global LOG
    models = ["gan", "vae-diffusion", "nf", "dm"]    
    results = []

    for model1 in models:
        for model2 in models:
            if model1 == "gan": continue
            elif model1 == model2:
                print(f"Reconstructing with {model2}")
                model = Model(model2, "cifar", use_thresholding=False)
                test_latents, range_ = get_latents(model2, "cifar", train=False, synthetic=False)
                for i, seed in enumerate(range_):
                    img = model.decode(torch.Tensor(test_latents[i]))
                    model.save_image(img, f"mapped/{seed}_{model2}.jpg")
                latent_mse = 0

            else:
                print(f"Testing {model1} -> {model2} mapping")
                model = Model(model2, "cifar", use_thresholding=True)
                latent_mse = test_exact_latent_space_mapping(model1, model, on_train_set=False, synthetic=False, save_images=True)

                if model2 == "gan":
                    test_exact_latent_space_mapping(model1, model, on_train_set=False, synthetic=True, save_images=True)
                
            results.append(
                {
                    "Encoder": model1,
                    "Decoder": model2,
                    "Latent MSE": latent_mse
                }
            )

    LOG.close()

    df = pd.DataFrame(results)
    df.to_pickle("cifar_results.pkl")
    df = pd.read_pickle("cifar_results.pkl").dropna()
    model_to_name = {
        'vae-diffusion': 'VAE',
        'vqvae': 'VQVAE',
        'nf': 'NF',
        'dm': 'DM',
    }

    df['Encoder'] = df['Encoder'].apply(lambda x: model_to_name[x])
    df['Decoder'] = df['Decoder'].apply(lambda x: model_to_name[x])

    print(df)
    plt.figure(figsize=(1.5, 1.5))
    df.Encoder=pd.Categorical(df.Encoder,categories=df.Encoder.unique(),ordered=True)
    df.Decoder=pd.Categorical(df.Decoder,categories=df.Decoder.unique(),ordered=True)
    pivoted = df.pivot(index="Encoder", columns="Decoder", values="Latent MSE")

    sns.heatmap(pivoted, annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.savefig(f"plots/latent_mse_cifar.png", dpi=300)

class ModelLite():
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

def test_celeba_mappings(from_saved=True):
    models = ["random", "gan", "vae-diffusion", "vqvae", "nf", "dm"] 

    if not from_saved:   
        results = []

        for model1 in models:
            for model2 in models:
                if model1 == "gan": continue
                if model2 == "random": continue
                elif model1 == model2:
                    print(f"Reconstructing with {model2}")
                    model = Model(model2, "celeba")
                    test_latents, range_ = get_latents(model2, "celeba", train=False, synthetic=False)
                    for i, seed in enumerate(range_):
                        img = model.decode(torch.Tensor(test_latents[i]))
                        model.save_image(img, f"mapped/{seed}_{model2}.jpg")
                    latent_mse = 0

                else:
                    print(f"Testing {model1} -> {model2} mapping")
                    model = ModelLite(model2, "celeba")
                    latent_mse_train = test_exact_latent_space_mapping(model1, model, on_train_set=True, synthetic=model2=="gan", save_images=False, regularized=model1 in ['dm', 'nf'])
                    latent_mse_test = test_exact_latent_space_mapping(model1, model, on_train_set=False, synthetic=model2=="gan", save_images=False, regularized=model1 in ['dm', 'nf'])
      
                results.append(
                    {
                        "Encoder": model1,
                        "Decoder": model2,
                        "Latent MSE Train": latent_mse_train,
                        "Latent MSE Test": latent_mse_test
                    }
                )

        LOG.close()

        df = pd.DataFrame(results)
        df.to_pickle("celeba_results.pkl")


    df = pd.read_pickle("celeba_results.pkl").dropna()

    model_to_name = {
        'random': 'Rand.',
        'gan': "GAN",
        'vae-diffusion': 'VAE',
        'vqvae': 'VQVAE',
        'nf': 'NF',
        'dm': 'DM',
    }

    df['Encoder'] = df['Encoder'].apply(lambda x: model_to_name[x])
    df['Decoder'] = df['Decoder'].apply(lambda x: model_to_name[x])

    print(df)
    plt.figure(figsize=(2.0, 1.7))
    df.Encoder=pd.Categorical(df.Encoder,categories=df.Encoder.unique(),ordered=True)
    df.Decoder=pd.Categorical(df.Decoder,categories=df.Decoder.unique(),ordered=True)
    
    pivoted = df.pivot(index="Decoder", columns="Encoder", values="Latent MSE Train")
    sns.heatmap(pivoted, annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title("Latent MSE (Train)")
    plt.savefig(f"plots/latent_mse_train_celeba.png", dpi=300)

    plt.figure(figsize=(2.0, 1.7))
    pivoted = df.pivot(index="Decoder", columns="Encoder", values="Latent MSE Test")
    sns.heatmap(pivoted, annot=True, fmt=".3f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title("Latent MSE (Test)")
    plt.savefig(f"plots/latent_mse_test_celeba.png", dpi=300)

def reconstruct_images(model, n_images):
    for i in range(1, n_images + 1):
        img = PIL.Image.open(f"data/celeba/00000{i}.jpg")
        img = transforms.CenterCrop(160)(img)
        img = transforms.Resize(size=64, antialias=True)(img)
        img = transforms.ToTensor()(img).unsqueeze(0)
        model.reconstruct(img, f"test_{i}.jpg")
        
if __name__ == "__main__":
    # train_exact_latent_space_mapping("dm", "vqvae", "celeba", synthetic=False, regularized=False)
    # train_exact_latent_space_mapping("nf", "dm", "celeba", synthetic=False, regularized=False)
    
    test_celeba_mappings()
    # test_cifar_mappings()
    
