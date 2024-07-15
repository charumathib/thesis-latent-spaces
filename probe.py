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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score
from joblib import dump
from tueplots import figsizes, fonts, bundles

plt.rcParams.update(bundles.icml2024())

TEST_SIZE = 100
LOG = None

class SoftmaxProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxProbe, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
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
        self.all_latents = []

        if split == "train":
            for i in range(10):
                self.all_latents += [(torch.load(f'latents/{model}/cifar_real/{str(idx).zfill(4)}.pth', map_location='cpu').detach().flatten().squeeze(), self.classes[i]) for idx in range(i * 500 + 1, (i * 500) + 401)]
        else:
            for i in range(10):
                self.all_latents += [(torch.load(f'latents/{model}/cifar_real/{str(idx).zfill(4)}.pth', map_location='cpu').detach().flatten().squeeze(), self.classes[i]) for idx in range((i * 500) + 401, (i * 500) + 411)]

        random.Random(7).shuffle(self.all_latents)
    def __len__(self):
        return len(self.all_latents)

    def __getitem__(self, idx):
        return self.all_latents[idx]
    
class LatentFiles():
    def __init__(self, model, feature, split, silent=False):
        print(model)
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
    #  optimizer = torch.optim.Adam(probe.parameters(), lr=0.0001, weight_decay=0.01) nf, dm
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.0001, weight_decay=0.001) # vae-diffusion

    for epoch in range(20):
        epoch_loss = 0
        for i, (z, label) in enumerate(trainloader):
            optimizer.zero_grad()
            output = probe(z)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} Loss: {epoch_loss}")
    
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
    torch.save(probe, f"probes/{model}_cifar.pth")

df = pd.read_csv('list_attr_celeba.csv')
# CelebA
def get_latents_with_attribute(model, attribute, split):
    TOTAL_LATENTS = 9100

    with_feature = df[df[attribute] == 1]['image_id']
    with_feature = list(filter(lambda x: int(x.split('.')[0]) < TOTAL_LATENTS, with_feature))
    without_feature = df[df[attribute] == -1]['image_id']
    without_feature = list(filter(lambda x: int(x.split('.')[0]) < TOTAL_LATENTS, without_feature))

    # we want to train on an equal number of positive and negative examples
    n_to_keep = min(len(with_feature), len(without_feature))
    train_size = int(0.8 * n_to_keep) # 0.8
    print(n_to_keep - train_size)
    test_size = n_to_keep - train_size
    # train_size = self.n_to_keep

    assert len(with_feature) - test_size >= train_size
    assert len(without_feature) - test_size >= train_size

    if split == "train":
        with_feature = with_feature[:train_size]
        without_feature = without_feature[:train_size]
    elif split == "test":
        with_feature = with_feature[len(with_feature) - test_size:]
        without_feature = without_feature[len(without_feature) - test_size:]
    elif split == "all":
        pass
    
    log(f"{attribute} {split} dataset")
    
    latents_with_feature = [torch.load(f'latents/{model}/celeba_real/{str(int(idx.split(".")[0])).zfill(6)}.pth').detach().flatten().squeeze() for idx in with_feature]
    latents_without_feature = [torch.load(f'latents/{model}/celeba_real/{str(int(idx.split(".")[0])).zfill(6)}.pth').detach().flatten().squeeze() for idx in without_feature]
    all_latents = latents_with_feature + latents_without_feature

    labels_with_feature = [1 for _ in with_feature]
    labels_without_feature = [0 for _ in without_feature]
    all_labels = labels_with_feature + labels_without_feature

    random.Random(7).shuffle(all_latents)
    random.Random(7).shuffle(all_labels)

    # Convert latents to numpy array
    all_latents = np.array([latent.numpy() for latent in all_latents])
    all_labels = np.array(all_labels)

    return all_latents, all_labels

# synthetic celeba latents
def get_latents(model_name, train):
    range_ = range(1, 9001) if train else range(9001, 9101)
    latents = [torch.load(f"latents/{model_name}/celeba_gen/{str(i).zfill(6)}.pth").flatten().detach() for i in range_]
    return torch.stack(latents, 0).numpy()

# CelebA
def train_probe(model_name, attribute):
    train_latents, train_labels = get_latents_with_attribute(model_name, attribute, "train")
    test_latents, test_labels = get_latents_with_attribute(model_name, attribute, "test")
    
    if model_name == "vae-diffusion" or model_name == "vqvae":
        alpha = 0.005
    elif model_name == "dm":
        alpha = 0.02
    elif model_name == "nf" or model_name in ["001001", "006001", "011001", "016001", "021001", "026001"]:
        alpha = 0.1

    probe = Lasso(alpha=alpha)
    probe.fit(train_latents, train_labels)

    # Compute accuracy on the train set
    train_predictions = probe.predict(train_latents)
    train_predictions = np.round(train_predictions)  # Round predictions to 0 or 1
    train_accuracy = accuracy_score(train_labels, train_predictions)
    log(f'Accuracy of the network on the train set: {100 * train_accuracy}%')

    # Compute accuracy on the test set
    test_predictions = probe.predict(test_latents)
    test_predictions = np.round(test_predictions)  # Round predictions to 0 or 1
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(len(test_labels))
    log(f'Accuracy of the network on the test set: {100 * test_accuracy}%')

    dump(probe, f'probes/{model_name}_{attribute}_probe.joblib')

def class_accuracies_cifar(model, name):
    test = LatentFilesCifar(model, "test")
    testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    probe = torch.load(f"probes/{model}_cifar.pth")
    assignments = [[0 for _ in range(10)] for _ in range(10)]

    correct = 0
    with torch.no_grad():
        for _, (z, label) in enumerate(testloader):
            outputs = probe(z)
            _, predicted = torch.max(outputs, axis=0)
            assignments[int(predicted.item())][int(label.item())] += 1
            # compute accuracy
            if int(predicted.item()) == int(label.item()):
                correct += 1
    
    print(f"Accuracy: {correct/100}")
    # # create a heatmap of assignments
    # df = pd.DataFrame(assignments)
    # df.index = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # df.columns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(df)
    # plt.figure(figsize=(2.2, 2.2))
    # a = sns.heatmap(df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    # plt.xlabel("True Label")
    # plt.ylabel("Predicted Label")
    # a.set_xticklabels(a.get_xmajorticklabels(), fontsize = 8)
    # a.set_yticklabels(a.get_ymajorticklabels(), fontsize = 8)
    # # plt.title(f"{name} Confusion Matrix")
    # plt.savefig(f"plots/{model}_cifar_class_accuracies.png", dpi=300)

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

TEST_SIZE = 100
def post_map_probe(from_, to_, attribute, gen_image=False, map_type="linear"):
    map = load(f"pickles/latent_mapping_{from_}_{to_}_celeba_{map_type}{'_reg' if from_ in ['nf', 'dm'] else ''}.joblib")
    probe = load(f"probes/{to_}_{attribute}_probe.joblib")
    
    log(f"Attribute: {attribute}")
    if not gen_image:
        from_latents, labels = get_latents_with_attribute(from_, attribute, "test")
        to_latents, _ = get_latents_with_attribute(to_, attribute, "test")
        
        to_latents_pred = map.predict(from_latents)
        probe_ = probe.predict(to_latents)
        probe_hat_ = probe.predict(to_latents_pred)

        acc_probe = accuracy_score(labels, np.round(probe_))
        acc_probe_hat = accuracy_score(labels, np.round(probe_hat_))
        log(f"Accuracy of {to_} probe: {acc_probe}")
        log(f"Accuracy of {from_} --> {to_} probe: {acc_probe_hat}")w
        log(f"Percent Delta Accuracy: {(acc_probe_hat - acc_probe)/acc_probe * 100}") # we want this to be positive
        log(f"Percentage matches: {(np.round(probe_) == np.round(probe_hat_)).sum().item()/TEST_SIZE}")

    else:
        from_latents = get_latents(from_, False)
        to_latents = get_latents(to_, False)

        to_latents_pred = map.predict(from_latents)
        probe_ = probe.predict(to_latents)
        probe_hat_ = probe.predict(to_latents_pred)
        log(f"Percentage matches: {(np.round(probe_) == np.round(probe_hat_)).sum().item()/TEST_SIZE}")

if __name__ == '__main__':
    attributes = ["Bald", "Blond_Hair", "Wavy_Hair", "Smiling", "Eyeglasses", "Heavy_Makeup", "Male", "No_Beard", "Pale_Skin", "Young"]
    models = ['nf', 'dm', 'vae-diffusion', 'vqvae', 'gan']
    
    for model in models:
        for attribute in attributes:
            train_probe(model, attribute)

    for model1 in models:
        for model2 in models:
            if model1 != model2 and not ((model1 == "nf" and model2 == "dm") or (model1 == "dm" and model2 == "nf")) and model2 != 'gan':
                LOG = open(f"probe_map_probe_logs_{model1}_{model2}.txt", "w")
                for attribute in attributes:
                    post_map_probe(model1, model2, attribute, gen_image=model1 == 'gan', map_type="linear")