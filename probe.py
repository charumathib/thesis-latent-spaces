from run import Model
import torch
import pandas as pd
import torch.nn as nn
import PIL.Image
from torchvision import transforms
import random
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from collections import defaultdict
from joblib import load
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns

TEST_SIZE = 100
LOG = None

class SoftmaxProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxProbe, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x) # relu
        x = self.linear2(x)
        x = self.softmax(x)

        return x.squeeze()
class Probe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Probe, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x) # relu
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x.squeeze()

def log(msg):
    print(msg)
    LOG.write(msg + "\n")

def compare_concepts(dataloaders, models, attributes):
    rounded_compositions_tensors = []
    summed_tensors = []
    summed_tensors_rounded = []

    # loop through the test dataset
    with torch.no_grad():
        for i, dataloader in enumerate(dataloaders):
            img_compositions = []
            for attribute in attributes:
                probe = torch.load(f"probes/{attribute}_{models[i]}_celeba.pth")
                for _, z in enumerate(dataloader):
                    output = probe(z)
                    img_compositions.append(output)
        
            stacked = torch.stack(img_compositions, -1)
            rounded_compositions_tensors.append(torch.round(stacked))
            summed_tensors.append(torch.sum(stacked, 0))
            summed_tensors_rounded.append(torch.sum(torch.round(stacked), 0))


    print("Instance Level Comparison")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            print(models[i], models[j]) 
            and_ = torch.logical_and(rounded_compositions_tensors[i], rounded_compositions_tensors[j])
            or_ = torch.logical_or(rounded_compositions_tensors[i], rounded_compositions_tensors[j])
            print(torch.mean(torch.sum(and_, axis=1)/torch.sum(or_, axis=1)).item())

    print("Dataset Level Comparison")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            print(models[i], models[j])
            # print the KL divergence of the two tensors
            print(entropy(summed_tensors[i], summed_tensors[j]))
    
    print("Dataset Level Comparison (Rounded)")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            print(models[i], models[j])
            # print the KL divergence of the two tensors
            print(entropy(summed_tensors_rounded[i], summed_tensors_rounded[j]))
            

def generate_latent_pickles(model, idx_start, idx_end, classes=[''], gan=False):
    ct = idx_start * 10
    for class_ in classes:
        for idx in range(idx_start, idx_end):
            img = PIL.Image.open(f"{'gen/gan' if gan else 'data'}/{model.dataset}/{class_ + '/' if class_ else ''}{str(idx).zfill(4 if model.dataset == 'cifar' else 6)}.jpg").convert('RGB')
            # img = PIL.Image.open(f"mapped/celeba/{idx}_vqvae_to_gan_w+_linear.jpg").convert('RGB')
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = transforms.Resize(model.pixel_shape[2], antialias=True)(img)
            img = transforms.CenterCrop(model.pixel_shape[2])(img)
            z = model.encode(img)

            # latent = model.generate(idx)
            
            torch.save(z, f'latents/{model.name}/{model.dataset}-real/{ct}_z.pth')
            # torch.save(z, f'latents/{model.name}/{model.dataset}-real/{str(idx).zfill(6)}_z.pth')
            # torch.save(z, f'latents/{model.name}/{model.dataset}-real/{str(idx).zfill(6)}_z.pth')
            # torch.save(z, f'latents/gan/{model.dataset}-real/{str(idx).zfill(6)}_z.pth')
            ct += 1

class LatentFilesNoAttr():
    def __init__(self, model):
        self.all_latents = [torch.load(f'latents/{model}/celeba-real/{str(idx).zfill(6)}_z.pth').detach().flatten().squeeze() for idx in range(14500, 15000)]
        random.Random(7).shuffle(self.all_latents)
    
    def __len__(self):
        return len(self.all_latents)

    def __getitem__(self, idx):
        return self.all_latents[idx]

class LatentFilesCifar():
    def __init__(self, model, split):
        # self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        if split == "train":
            self.all_latents = [(torch.load(f'latents/{model["name"]}/cifar-real/{idx}_z.pth').detach().flatten().squeeze(), self.classes[idx // 195]) for idx in range(1950)]
            random.Random(7).shuffle(self.all_latents)
        else:
            self.all_latents = [(torch.load(f'latents/{model["name"]}/cifar-real/{idx}_z.pth').detach().flatten().squeeze(), (idx - 1950) // 5) for idx in range(1950, 2000)]
    
    def __len__(self):
        return len(self.all_latents)

    def __getitem__(self, idx):
        return self.all_latents[idx]
    
class LatentFiles():
    def __init__(self, model, feature, split, silent=False):
        print(model)
        if 'vae' in model:
            self.total_latents = 15000
        elif model == "nf-celeba":
            self.total_latents = 10000
        
        self.test_size = 100

        df = pd.read_csv('list_attr_celeba.csv')
        self.split = split

        with_feature = list(df[df[feature] == 1]['image_id'])
        with_feature = list(filter(lambda x: int(x.split('.')[0]) < self.total_latents, with_feature))
        without_feature = df[df[feature] == -1]['image_id']
        without_feature = list(filter(lambda x: int(x.split('.')[0]) < self.total_latents, without_feature))

        # we want to train on an equal number of positive and negative examples
        self.n_to_keep = min(len(with_feature), len(without_feature))
        train_size = int(0.8 * self.n_to_keep)
        # train_size = self.n_to_keep

        if split == "train":
            with_feature = with_feature[:train_size]
            without_feature = without_feature[:train_size]
        elif split == "test":
            with_feature = with_feature[len(with_feature) - self.test_size:]
            without_feature = without_feature[len(without_feature) - self.test_size:]
            # print(min(with_feature))
            # print(min(without_feature))
        elif split == "all":
            pass
        
        if not silent:
            log(f"{feature} {split} dataset")
            
        self.latents_with_feature = [(torch.load(f'latents/{model}/celeba-real/{str(int(idx.split(".")[0])).zfill(6)}_z.pth').detach().flatten().squeeze(), 1) for idx in with_feature]
        self.latents_without_feature = [(torch.load(f'latents/{model}/celeba-real/{str(int(idx.split(".")[0])).zfill(6)}_z.pth').detach().flatten().squeeze(), 0) for idx in without_feature]
        self.all_latents = self.latents_with_feature + self.latents_without_feature
        random.Random(7).shuffle(self.all_latents)

        if not silent:
            log(f"Dataset size: {len(self.all_latents)}")

    def __len__(self):
        return len(self.all_latents)

    def __getitem__(self, idx):
        return self.all_latents[idx]

def train_probe_cifar(model):
    train = LatentFilesCifar(model, "train")
    test = LatentFilesCifar(model, "test")
    trainloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)

    latent_shape = train[0][0].shape[0]
    probe = SoftmaxProbe(latent_shape, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.0001)

    for epoch in range(100):
        for i, (z, label) in enumerate(trainloader):
            optimizer.zero_grad()
            output = probe(z)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    
    # print accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (z, label) in enumerate(trainloader):
            outputs = probe(z)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    log(f'Accuracy of the network on the {total} train images: {100 * correct / total}%')

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (z, label) in enumerate(testloader):
            outputs = probe(z)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    log(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')
    torch.save(probe, f"probes/{model['name']}_cifar.pth")

def train_probe(model, attribute):
    train = LatentFiles(model["name"], attribute, "train")
    test = LatentFiles(model["name"], attribute, "test")
    
    trainloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)

    probe = Probe(128, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001) # for VAE the LR was 0.001 and for most attributes for VQVAE was 0.0001 for most attr for NF (except straight hair 0.000001) was 0.00001
    for epoch in range(20): # 20
        for i, (z, label) in enumerate(trainloader):
            optimizer.zero_grad()
            output = probe(z)
            # print(output)
            # print(label.float())
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
    # print accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (z, label) in enumerate(trainloader):
            outputs = probe(z)
            predicted = torch.round(outputs)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    log(f'Accuracy of the network on the {total} train images: {100 * correct / total}%')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (z, label) in enumerate(testloader):
            outputs = probe(z)
            predicted = torch.round(outputs)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    log(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')
    
    # torch.save(probe.state_dict(), f"probes/{attribute}_{model.model}_celeba_statedict.pkl")
    torch.save(probe, f"probes/{attribute}_{model['name']}_celeba.pth")

def class_accuracies_cifar(model):
    test = LatentFilesCifar(model, "test")
    testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    probe = torch.load(f"probes/{model['name']}_cifar.pth")
    assignments = [[0 for _ in range(10)] for _ in range(10)]

    with torch.no_grad():
        for _, (z, label) in enumerate(testloader):
            outputs = probe(z)
            _, predicted = torch.max(outputs, axis=0)
            assignments[int(predicted.item())][int(label.item())] += 1
    
    # create a heatmap of assignments
    df = pd.DataFrame(assignments)
    df.index = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    df.columns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(df)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"{model['model'].upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"plots/{model['name']}_cifar_class_accuracies.png", dpi=300)

def post_map_probe_cifar(from_, to_, to_latent_layer, gen_image=False, load_gen_img_from_file=False, map_type="linear"):
    map = load(f"pickles/latent_mapping_{from_['name']}_{to_['name']}_cifar_{to_latent_layer}_{map_type}.joblib")
    probe = torch.load(f"probes/{to_['name']}_cifar.pth")
    
    if not gen_image:
        from_latents = LatentFilesCifar(from_, "test")
        to_latents = LatentFilesCifar(to_, "test")

        from_loader = torch.utils.data.DataLoader(from_latents, batch_size=5, shuffle=False)
        to_loader = torch.utils.data.DataLoader(to_latents, batch_size=5, shuffle=False)

        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        with torch.no_grad():
            for i, (batch_from, z) in enumerate(zip(cycle(from_loader), to_loader)):
                log(f"Attribute: {classes[i]}")
                z_hat = torch.Tensor(map.predict(batch_from[0]))
                
                true_labels = batch_from[1]
                assert(torch.all(true_labels == z[1]))
                
                probe_ = probe(z[0])
                probe_hat_ = probe(z_hat)

                log(f"KL Divergence: {torch.nn.functional.kl_div(torch.log(probe_), probe_hat_).item()}")
    else:
        kl_divs = []
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        belonged_classes = [
            'ship', 'dog', 'bird', 'truck', 'ship', 'frog', 'horse', 'automobile', 'dog', 'ship', 'automobile', 'cat',
            'ship', 'airplane', 'frog', 'truck', 'horse', 'ship', 'cat', 'deer', 'cat', 'automobile', 'cat', 'bird',
            'automobile', 'frog', 'dog', 'airplane', 'dog', 'airplane', 'ship', 'truck', 'ship', 'airplane', 'frog',
            'ship', 'airplane', 'dog', 'frog', 'bird', 'automobile', 'frog', 'bird', 'deer', 'dog', 'cat', 'deer',
            'ship', 'automobile', 'airplane'
        ]

        kl_divs = [[] for _ in range(10)]
        for idx in range(1950, 2000):
            from_latent = torch.load(f"latents/{from_['name']}/cifar/{idx}_w+.pth").detach() # str(idx).zfill(6)
            z_to = torch.load(f"latents/{to_['name']}/cifar/{idx}_z.pth").detach().reshape(1, -1)
            z_hat = torch.Tensor(map.predict(from_latent.reshape(1, -1))).detach().reshape(1, -1)

            print(z_to.shape)
            print(z_hat.shape)
            probe_ = probe(z_to)
            probe_hat_ = probe(z_hat)
        
            kl_divs[classes.index(belonged_classes[idx-1950])].append(torch.nn.functional.kl_div(torch.log(probe_), probe_hat_).item())

        for i, class_ in enumerate(classes):
            log(f"Attribute: {class_}")
            log(f"KL Divergence: {np.mean(kl_divs[i])}")

TEST_SIZE = 200
def post_map_probe(from_, to_, to_latent_layer, attribute, gen_image=False, load_gen_img_from_file=False, map_type="linear"):
    map = load(f"pickles/latent_mapping_{from_['name']}_{to_['name']}_celeba_{to_latent_layer}_{map_type}.joblib")
    # map = torch.load(f"pickles/latent_mapping_{from_['name']}_{to_['name']}_celeba_{to_latent_layer}_linear.pth")
    probe = torch.load(f"probes/{attribute}_{to_['model']}_celeba.pth")
    pred_z_to = []
    pred_z_to_hat = []
    
    log(f"Attribute: {attribute}")
    if not gen_image:
        from_latents = LatentFiles(from_['name'], attribute, "test", silent=True)
        to_latents = LatentFiles(to_['name'], attribute, "test", silent=True)
        from_loader = torch.utils.data.DataLoader(from_latents, batch_size=TEST_SIZE, shuffle=False)
        to_loader = torch.utils.data.DataLoader(to_latents, batch_size=TEST_SIZE, shuffle=False)

        with torch.no_grad():
            batch_from = next(iter(from_loader))
            z_hat = torch.Tensor(map.predict(batch_from[0]))
            # z_hat = batch_from[0]
            z = next(iter(to_loader))
            
            true_labels = batch_from[1]
            assert(torch.all(true_labels == z[1]))
            
            probe_ = probe(z[0])
            probe_hat_ = probe(z_hat)
            
            # print accuracy
            log(f"# positive examples {torch.sum(true_labels)}")
            log(f"# negative examples {len(true_labels) - torch.sum(true_labels)}")
            acc_to = (np.round(probe_) == true_labels).sum().item()/TEST_SIZE
            log(f"Accuracy of {to_['model']} probe: {acc_to}")
            acc_from = (np.round(probe_hat_) == true_labels).sum().item()/TEST_SIZE
            log(f"Accuracy of {from_['model']} --> {to_['model']} probe: {acc_from}")
            log(f"Percent Delta Accuracy: {(acc_to - acc_from)/acc_to * 100}")
            
            log(f"MSE: {((probe_ - probe_hat_) ** 2).mean()}")
            log(f"Cosine Similarity: {cosine(u=probe_, v=probe_hat_)}")
            
            # number of agreement
            log(f"Percentage matches: {(np.round(probe_) == np.round(probe_hat_)).sum().item()/TEST_SIZE}")
    else:
        # used 5000 -> 5000 + TEST_SIZE for the VAE at various stages of training probing
        for idx in range(14800, 15000):
            if gen_image:
                # img = PIL.Image.open(f"gen/gan/celeba/{str(idx).zfill(6)}.jpg").convert('RGB')
                from_latent = torch.load(f"latents/{from_['name']}/celeba/{idx}_w+.pth").detach() # str(idx).zfill(6)
                z_to = torch.load(f"latents/{to_['name']}/celeba/{idx}_z.pth").detach().flatten()

            z_to_hat = torch.Tensor(map.predict(from_latent.reshape(1, -1)))

            pred_z_to.append(probe(z_to).item())
            pred_z_to_hat.append(probe(z_to_hat).item())
        
        pred_z_to = np.array(pred_z_to) 
        pred_z_to_hat = np.array(pred_z_to_hat)

        print(pred_z_to.shape)
        print(pred_z_to_hat.shape)
        log(f"MSE: {((pred_z_to - pred_z_to_hat) ** 2).mean()}")
        log(f"Cosine Similarity: {cosine(u=pred_z_to, v=pred_z_to_hat)}")

        # ones = np.round(pred_z_to) == np.ones_like(pred_z_to)
        # ones_hat = np.round(pred_z_to_hat) == np.ones_like(pred_z_to_hat)

        # unique, counts = np.unique(np.round(pred_z_to_hat) + np.round(pred_z_to), return_counts=True)
        # dict_ = defaultdict(int, zip(unique, counts))
        # print(dict_)

        # log(f"Percent Positive Matches: {dict_[2]/np.sum(ones) * 100}")
        # log(f"Percent Negative Matches: {dict_[0]/(end - np.sum(ones)) * 100}")
        log(f"Percent Matches: {(np.round(pred_z_to) == np.round(pred_z_to_hat)).sum().item()/TEST_SIZE * 100}")

def double_post_map_probe(from_, to_, attribute, start, end, gen_image=False, load_gen_img_from_file=False, map_type="linear"):
    map_1 = torch.load(f"pickles/latent_mapping_{to_.name}_{from_.name}_{to_.dataset}_simul.pth") #_{to_.latent_layer}_{map_type}
    map_2 = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_simul.pth")

    probe = torch.load(f"probes/{attribute}_{to_.model}_celeba.pth")
    pred_z_to = []
    pred_z_to_hat = []
    
    log(f"Attribute: {attribute}")
    # from_latents = LatentFiles(from_, attribute, "test", silent=True)
    to_latents = LatentFiles(to_, attribute, "test", silent=True)
    # from_loader = torch.utils.data.DataLoader(from_latents, batch_size=200, shuffle=False)
    to_loader = torch.utils.data.DataLoader(to_latents, batch_size=200, shuffle=False)

    with torch.no_grad():
        # batch_from = next(iter(from_loader))
        z = next(iter(to_loader))
        z_hat = map_2(map_1(z[0]))
        
        true_labels = z[1]
        assert(torch.all(true_labels == z[1]))
        
        probe_ = probe(z[0])
        probe_hat_ = probe(z_hat)
        
        # print accuracy
        log(f"# positive examples {torch.sum(true_labels)}")
        log(f"# negative examples {len(true_labels) - torch.sum(true_labels)}")
        acc_to = (np.round(probe_) == true_labels).sum().item()/200
        log(f"Accuracy of {to_.model} probe: {acc_to}")
        acc_from = (np.round(probe_hat_) == true_labels).sum().item()/200
        log(f"Accuracy of {from_.model} --> {to_.model} probe: {acc_from}")
        log(f"Percent Delta Accuracy: {(acc_to - acc_from)/acc_to * 100}")
        
        log(f"MSE: {((probe_ - probe_hat_) ** 2).mean()}")
        log(f"Cosine Similarity: {cosine(u=probe_, v=probe_hat_)}")
        
        # number of agreement
        log(f"Percentage matches: {(np.round(probe_) == np.round(probe_hat_)).sum().item()/200}")

        # pred_z_to.append(label)

def double_post_map_probe(from_, to_, attribute, start, end, gen_image=False, load_gen_img_from_file=False, map_type="linear"):
    map_1 = torch.load(f"pickles/latent_mapping_{to_.name}_{from_.name}_{to_.dataset}_simul.pth") #_{to_.latent_layer}_{map_type}
    map_2 = torch.load(f"pickles/latent_mapping_{from_.name}_{to_.name}_{to_.dataset}_simul.pth")

    probe = torch.load(f"probes/{attribute}_{to_.model}_celeba.pth")
    pred_z_to = []
    pred_z_to_hat = []
    
    log(f"Attribute: {attribute}")
    if not gen_image:
        # from_latents = LatentFiles(from_, attribute, "test", silent=True)
        to_latents = LatentFiles(to_, attribute, "test", silent=True)
        # from_loader = torch.utils.data.DataLoader(from_latents, batch_size=200, shuffle=False)
        to_loader = torch.utils.data.DataLoader(to_latents, batch_size=200, shuffle=False)

        with torch.no_grad():
            # batch_from = next(iter(from_loader))
            z = next(iter(to_loader))
            z_hat = map_2(map_1(z[0]))
            
            true_labels = z[1]
            assert(torch.all(true_labels == z[1]))
            
            probe_ = probe(z[0])
            probe_hat_ = probe(z_hat)
            
            # print accuracy
            log(f"# positive examples {torch.sum(true_labels)}")
            log(f"# negative examples {len(true_labels) - torch.sum(true_labels)}")
            acc_to = (np.round(probe_) == true_labels).sum().item()/200
            log(f"Accuracy of {to_.model} probe: {acc_to}")
            acc_from = (np.round(probe_hat_) == true_labels).sum().item()/200
            log(f"Accuracy of {from_.model} --> {to_.model} probe: {acc_from}")
            log(f"Percent Delta Accuracy: {(acc_to - acc_from)/acc_to * 100}")
            
            log(f"MSE: {((probe_ - probe_hat_) ** 2).mean()}")
            log(f"Cosine Similarity: {cosine(u=probe_, v=probe_hat_)}")
            
            # number of agreement
            log(f"Percentage matches: {(np.round(probe_) == np.round(probe_hat_)).sum().item()/200}")

            # pred_z_to.append(label)
    else:
        for idx in range(TEST_SIZE):
            if gen_image:
                img = PIL.Image.open(f"gen/gan/{to_.dataset}/{str(idx).zfill(6)}.jpg").convert('RGB')
                from_latent = torch.load(f"latents/{from_.name}/{to_.dataset}/{idx}_w+.pth").detach()
                z_to = to_.encode(img)
            else:
                # load the latents
                img = PIL.Image.open(f"data/{to_.dataset}/{str(idx).zfill(6)}.jpg").convert('RGB')
                
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = transforms.Resize(to_.pixel_shape[2], antialias=True)(img)
            img = transforms.CenterCrop(to_.pixel_shape[2])(img)

            if not gen_image:
                from_latent = from_.encode(img)

            z_to_hat = map(from_latent.flatten().squeeze().detach())

            pred_z_to.append(probe(z_to).item())
            pred_z_to_hat.append(probe(z_to_hat).item())
        
        pred_z_to = np.array(pred_z_to)
        pred_z_to_hat = np.array(pred_z_to_hat)

        print(pred_z_to.shape)
        print(pred_z_to_hat.shape)
        log(f"MSE: {((pred_z_to - pred_z_to_hat) ** 2).mean()}")
        log(f"Cosine Similarity: {cosine(u=pred_z_to, v=pred_z_to_hat)}")

        ones = np.round(pred_z_to) == np.ones_like(pred_z_to)
        ones_hat = np.round(pred_z_to_hat) == np.ones_like(pred_z_to_hat)

        print(sum(ones))
        print(sum(ones_hat))

        unique, counts = np.unique(np.round(pred_z_to_hat) + np.round(pred_z_to), return_counts=True)
        dict_ = defaultdict(int, zip(unique, counts))
        print(dict_)

        log(f"Percent Positive Matches: {dict_[2]/np.sum(ones) * 100}")
        log(f"Percent Negative Matches: {dict_[0]/(end - np.sum(ones)) * 100}")


if __name__ == '__main__':
    # attributes = list(pd.read_csv('list_attr_celeba.csv').columns[1:])
    pass
