import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from torchvision import transforms

from cifar_ae import create_model

import legacy

import json

from glow.datasets import get_CIFAR10, get_SVHN, postprocess, preprocess
from glow.model import Glow

device = torch.device("cpu")

class Model():
    def __init__(self, name, pickle, dataset):
        self.name = name
        self.dataset = dataset
        if self.name == "nf": # normalizing flow
            with open("glow/hparams.json") as json_file:  
                hparams = json.load(json_file)
                

            self.model_ = Glow((32, 32, 3), hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                        hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], 10,
                        hparams['learn_top'], hparams['y_condition'])

            self.model_.load_state_dict(torch.load(pickle, map_location=torch.device('cpu')))
            self.model_.set_actnorm_init()
            self.model_ = self.model_.to(device)
            self.model_ = self.model_.eval()
        elif self.name == "ae":
            self.model_ = create_model()
            self.model_.load_state_dict(torch.load(pickle, map_location='cpu'))
        elif self.name == "vae":
            self.model_ = torch.load(pickle, map_location='cpu')
        elif self.name == "gan":
            with open(pickle, 'rb') as f:
                self.model_ = legacy.load_network_pkl(f)['G_ema'].to(device)
        else:
            raise NotImplementedError

    def encode(self, img):
        if self.name == "nf":
            z, _, _ = self.model_(img)
            return z
        if self.name == "ae":
            return self.model_.encoder(img)
        elif self.name == "vae":
            mu, log_var = self.model_.encode(img)
            return self.model_.reparameterize(mu, log_var)
        else:
            raise NotImplementedError
    
    def partial_decode(self, latent):
        if self.name == "vae":
            return self.model_.decode_partial(latent)

        raise NotImplementedError
    
    def decode(self, latent):
        if self.name == "nf":
            return postprocess(self.model_(y_onehot=None, z=latent, temperature=0, reverse=True)).cpu()
        elif self.name == "ae":
            return self.model_.decoder(latent)
        elif self.name == "vae":
            return self.model_.decode(latent).view(1, 3, self.pixel_size(), self.pixel_size())
        elif self.name == "gan":
            # TODO: add a flag to this
            img = self.model_(latent, None, noise_mode='none', force_fp32=True)
            # img = self.model_.synthesis(latent, noise_mode='none', force_fp32=True)
            if self.pixel_size() != img.shape[2]:
                img = transforms.Resize(size=self.pixel_size(), antialias=True)(img)
            return img
        else:
            raise NotImplementedError
    
    def reconstruct(self, img, filename, n_recon=5) :
        if self.name == "nf":
            z = self.encode(img)
            print("##########")
            pic = self.decode(z).squeeze()
            plt.imshow(pic.permute(1,2,0))
            plt.show()
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
                img_size = self.pixel_size()
                pic = recon[0].view(1, 3, img_size, img_size)
                pics = torch.cat((pics, pic), dim=0)
                save_image(pic, f"rec/{filename}")
        else:
            raise NotImplementedError
        
        
        # save_image(pics, f"rec/{filename}", nrow=n_recon + 1)

    def generate_grid(self, start_seed, n_samples):
        pics = torch.empty(0, 3, self.pixel_size(), self.pixel_size())
        for i in range(n_samples**2):
            seed = start_seed + i
            _, img = self.generate(seed)
            pics = torch.cat((pics, img), dim=0)
        
        # TODO: fix color saturation for GAN
        save_image(pics, f"gen/{self.name}/{self.dataset}_grid.jpg", nrow=n_samples)

    def save_image(self, img, filename):
        if self.name == "ae" or self.name == "vae":
            save_image(img, filename)
        else:
            img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(filename)

    def generate(self, seed):
        if self.name == "ae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape())).to(torch.float).to(device)
            img = self.model_.decoder(latent)
            save_image(img, f'gen/{self.name}/{self.dataset}/{str(seed).zfill(6)}.jpg')
        elif self.name == "vae":
            latent = torch.from_numpy(np.random.RandomState(seed).randn(*self.latent_shape(self.dataset))).to(torch.float).to(device)
            img = self.model_.decode(latent).view(1, 3, self.pixel_size(), self.pixel_size())
            save_image(img, f'gen/{self.name}/{self.dataset}/{str(seed).zfill(6)}.jpg')
        elif self.name == "gan":
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.model_.z_dim)).to(torch.float32).to(device)
            latent = self.model_.mapping(z, None)
            img = self.model_.synthesis(latent, noise_mode='none', force_fp32=True)

            if self.pixel_size() != img.shape[2]:
                img = transforms.Resize(size=self.pixel_size(), antialias=True)(img)

            latent = z
            img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_[0].cpu().numpy(), 'RGB').save(f'gen/{self.name}/{self.dataset}/{str(seed).zfill(6)}.jpg')

        return latent, img
    
    def pixel_size(self):
        if self.dataset == "cifar":
            return 32
        elif self.dataset == "celeba" or self.dataset == "metfaces":
            return 150

    def latent_shape(self):
        if self.name == "ae":
            if self.dataset == "cifar":
                return (1, 48, 4, 4)
        elif self.name == "vae":
            if self.dataset == "celeba":
                return (1, 128)
        elif self.name == "gan":
            if self.dataset == "cifar":
                return (1, 8, 512)
            elif self.dataset == "celeba":
                return (1, 14, 512)
            elif self.dataset == "metfaces":
                return (1, 18, 512)
        raise NotImplementedError
    
    # returns the size of the latent space for the given model and dataset
    def latent_dim(self):
        if self.name == "ae":
            if self.dataset == "cifar":
                return 48 * 4 * 4
        elif self.name == "vae":
            if self.dataset == "celeba":
                return 128
        elif self.name == "gan":
            if self.dataset == "cifar":
                return 8 * 512
            elif self.dataset == "celeba":
                return 14 * 512
            elif self.dataset == "metfaces":
                return 18 * 512
        raise NotImplementedError

def train_latent_space_mapping(from_, to_, data, n_epochs, n_datapoints, from_saved=False, save_pickles=True, partial=False):
    print(f"Training latent space mapping from {from_.name} to {to_.name} over {n_datapoints} datapoints and {n_epochs} epochs")
   
    map = nn.Linear(from_.latent_dim(), to_.latent_dim() if not partial else 512)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(map.parameters(), lr=1e-4)

    # params for w mapping!!
    # optimizer = optim.SGD(map.parameters(), lr=0.01, momentum=0.9)

    # params for z mapping!!
    optimizer = optim.SGD(map.parameters(), lr=0.01)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for iter in range(n_datapoints):
            if save_pickles and (from_saved or epoch > 0):
                from_latent = torch.load(f"latents/{from_.name}/{from_.dataset}/{iter}.pth")
                to_latent = torch.load(f"latents/{to_.name}/{to_.dataset}/{iter}.pth")
            else:
                # use GAN to generate the data
                if data == None:
                    if from_.name == "gan":
                        from_latent, img = from_.generate(iter)
                        to_latent = to_.encode(img)
                    elif to_.name == "gan":
                        to_latent, img = to_.generate(iter)
                        from_latent = from_.encode(img)
                else:
                    if from_.dataset == "cifar":
                        img = data[iter][0].reshape(1, 3, 32, 32)
                    else:
                        raise NotImplementedError

                    from_latent = from_.encode(img)
                    to_latent = to_.encode(img)
                
                # if save_pickles:
                #     # save the latent representations for later
                #     torch.save(from_latent, f"latents/{from_.name}/{from_.dataset}/{iter}.pth")
                #     torch.save(to_latent, f"latents/{to_.name}/{to_.dataset}/{iter}.pth")
                
            pred_to_latent = map(from_latent.flatten())

            # loss = criterion(pred_to_latent, to_.model_.mapping(to_latent, None).flatten())
            loss = criterion(pred_to_latent, to_latent.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if iter % 100 == 0:
            #     print(f"Epoch {epoch} Iter {iter} Loss: {loss.item()}")

        losses.append(epoch_loss/n_datapoints)
        print(f"Epoch {epoch} Loss: {epoch_loss/n_datapoints}")

    # plt.scatter(range(n_epochs), losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title(f"[{from_.dataset}] {from_.name} to {to_.name} latent space mapping")
    # plt.savefig(f"plots/{from_.dataset}_{from_.name}_{to_.name}.png")
    # plt.show()

    torch.save(map, f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}.pth")

# TODO: refactor this code
def train_latent_space_mapping_two(from_, to_, img, n_epochs):   
    map = nn.Linear(from_.latent_dim(), 512)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(map.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        from_latent = from_.encode(img)
        pred_to_latent = map(from_latent.flatten())

        loss = criterion(img, to_.decode(pred_to_latent.view(1, 512)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"[{from_.dataset}] {from_.name} to {to_.name} latent space mapping")
    plt.savefig(f"plots/{from_.dataset}_{from_.name}_{to_.name}.png")
    plt.show()

    torch.save(map, f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}.pth")

def test_latent_space_mapping_two(from_, to_, data, start, end):
    map = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}.pth")
    for pic in ["00000.jpg"]:
        img = PIL.Image.open(f"data/celebahq/{pic}").convert('RGB')
        w, h = img.size
        s = min(w, h)
        # img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((150, 150), PIL.Image.LANCZOS)
        img = np.array(img, dtype=np.uint8)
        img = torch.from_numpy(img.transpose([2, 0, 1])).reshape((1, 3, 150, 150)).to(torch.float).to(device)
        
        from_latent = from_.encode(img)
        from_decoded = from_.decode(from_latent)

        to_latent = map(from_latent)
        to_decoded = to_.decode(to_latent)


        from_.save_image(from_decoded, f"test/{pic}_{from_.name}.jpg")
        from_.save_image(to_decoded, f"test/{pic}_{to_.name}.jpg")

def test_latent_space_mapping(from_, to_, data, start, end):
    for seed in range(start, end):
        map = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}.pth")

        if from_.name == "gan":
            from_latent, img = from_.generate(seed)
            to_latent = to_.encode(img)
        elif to_.name == "gan":
            to_latent, img = to_.generate(seed)
            from_latent = from_.encode(img)
        else:
            # TODO: this is only implemented for cifar
            assert data is not None
            img = data[seed][0].reshape(1, 3, 32, 32)
            from_latent = from_.encode(img)
            to_latent = to_.encode(img)

        from_decoded = from_.decode(from_latent)
        to_decoded = to_.decode(to_latent)

        pred_to_latent = map(from_latent.flatten())
        # to_decoded_pred = to_.model_.synthesis(pred_to_latent.view(to_.latent_shape()), force_fp32=True)
        to_decoded_pred = to_.decode(pred_to_latent.view(1, 512))

        from_.save_image(from_decoded, f"test/{seed}_{from_.name}.jpg")
        to_.save_image(to_decoded, f"test/{seed}_{to_.name}.jpg")
        to_.save_image(to_decoded_pred, f"test/{seed}_{from_.name}_to_{to_.name}_z.jpg")

        # from_.save_image(from_decoded, f"mapped/{to_.dataset}/{seed}_{from_.name}.jpg")
        # to_.save_image(to_decoded, f"mapped/{to_.dataset}/{seed}_{to_.name}.jpg")
        # to_.save_image(to_decoded_pred, f"mapped/{to_.dataset}/{seed}_{from_.name}_to_{to_.name}.jpg")

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
            # TODO: this is only implemented for cifar
            assert data is not None
            img = data[seed][0].reshape(1, 3, 32, 32)
            from_latent = from_.encode(img)

        pred_to_latent = map1(from_latent.flatten())
        pred_from_latent = map2(pred_to_latent)

        from_decoded_pred = from_.decode(pred_from_latent.view(from_.latent_shape()))

        from_.save_image(from_decoded_pred, f"mapped/{to_.dataset}/{seed}_{from_.name}_to_{to_.name}_to_{from_.name}.jpg")

if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), ])
    transform = transforms.Compose([transforms.ToTensor(), preprocess])
    data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    
    # gan = Model("gan", "pickles/cifar-stylegan.pkl", "cifar")
    # gan = Model("gan", "pickles/celebahq-256.pkl", "celeba")
    # gan = Model("gan", "pickles/metfaces.pkl", "metfaces")

    # vae = Model("vae", "pickles/celeba-vae.pth", "celeba")
    
    nf = Model("nf", "pickles/glow_affine_coupling.pt", "cifar")
    plt.imshow(data[1][0].permute(1, 2, 0))
    plt.show()
    nf.reconstruct(data[1][0].unsqueeze(0), "rec.png")

    '''
    img = PIL.Image.open(f"data/celebahq/00000.jpg").convert('RGB')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((150, 150), PIL.Image.LANCZOS)
    img = np.array(img, dtype=np.uint8)
    img = torch.from_numpy(img.transpose([2, 0, 1])).reshape((1, 3, 150, 150)).to(torch.float).to(device)
    '''
    # train_latent_space_mapping_two(vae, gan, img, 100)
    # train_latent_space_mapping(vae, gan, None, 50, 2000, from_saved=True, save_pickles=True, partial=True)

    # ae1 = Model("ae", "pickles/cifar-ae-1.pkl")
    # ae2 = Model("ae", "pickles/cifar-ae-2.pkl")

    # vae.generate_grid(0, "celeba", 10)
    # gan.generate_grid(0, "cifar", 10)
    # gan2.generate_grid(0, "celeba", 10)
    # ae1.generate_grid(0, "cifar", 10)

    # for seed in range(10):
    #     # latent, img = gan.generate(seed, "celeba")
    #     img = PIL.Image.open(f"data/celebahq/0000{seed}.jpg").convert('RGB')
    #     w, h = img.size
    #     s = min(w, h)
    #     # img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    #     img = img.resize((150, 150), PIL.Image.LANCZOS)
    #     img = np.array(img, dtype=np.uint8)
    #     img = torch.from_numpy(img.transpose([2, 0, 1])).reshape((1, 3, 150, 150)).to(torch.float).to(device)
    #     # print(img.shape)
    #     vae.reconstruct(img, f"vae_{seed}.jpg", n_recon=1)
    #     # ae1.reconstruct(data[seed][0].reshape(1, 3, 32, 32), "cifar", f"ae_{seed}.jpg", n_recon=1)

    # train_latent_space_mapping(ae1, ae2, "cifar", data, 5, 5000, from_saved=False, save_pickles=False)
    # test_latent_space_mapping(ae1, ae2, "cifar", data, 5000, 5010)
    # train_latent_space_mapping(gan, vae, None, 2, 2000, from_saved=True, save_pickles=True)
    # test_double_latent_space_mapping(gan, vae, None, 10, 20)

    # test_latent_space_mapping(vae, gan, None, 0, 20)
    # test_latent_space_mapping_two(vae, gan, None, 0, 1)

    # test_latent_space_mapping(gan, vae, None, 0, 20)
     
        

        
