import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tueplots import bundles

plt.rcParams.update(bundles.icml2024())
plt.rcParams["text.usetex"] = True

accuracies = {"vae-diffusion": [], "vqvae": [],  "nf": [], 'dm': []}
per_attr_acc = []
data = []


model_to_name = {
    "vae-diffusion": "VAE",
    "nf": "NF",
    "dm": "DM",
    "vqvae": "VQVAE"
}
def parse_training_accuracies(models):
    data = []
    attributes = ["Bald", "Blond_Hair", "Wavy_Hair", "Smiling", "Eyeglasses", 
                  "Heavy_Makeup", "Male", "No_Beard", "Pale_Skin", "Young"]
    
    for model in models:
        with open (f"probe_log_{model}_celeba.txt", "r") as file:
            lines = file.readlines()

        for i in range(0, len(lines), 4):
            attribute = lines[i].split(" ")[0].strip()
            if attribute not in attributes:
                continue
            data.append({
                "Attribute": attribute,
                "Latents": model_to_name[model],
                "Test Accuracy": float(lines[i+3].split(": ")[1].strip()[:-1])
            })
    
    df = pd.DataFrame(data)
    print(df)
    plt.figure(figsize=(10, 0.7))
    a = sns.heatmap(df.pivot(index="Latents", columns="Attribute", values="Test Accuracy"), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    a.set_xticklabels([])
    a.tick_params(bottom=False)
    a.set_yticklabels(a.get_ymajorticklabels(), fontsize = 8, rotation=0)
    a.set_xlabel("")
    a.set_ylabel(a.get_ylabel(), fontsize = 8)
    plt.savefig(f"plots/test_accuracy_heatmap_training.png", dpi=400)
    
def parse_post_map_results_cifar(models):
    data = []

    for model1 in models:
        for model2 in models:
            if model1 != model2:
                with open (f"post_map_probe_logs_{model1}_{model2}_cifar.txt", "r") as file:
                    lines = file.readlines()
                
                for i in range(0, len(lines), 2):
                    data.append({
                        "Attribute": lines[i].split(": ")[1].strip(),
                        "KL Divergence": float(lines[i+1].split(": ")[1].strip()),
                        "Latents": f"{model1} --> {model2}"
                    })

    for model in models:
        with open (f"post_map_probe_logs_gan_{model}_cifar.txt", "r") as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                data.append({
                    "Attribute": lines[i].split(": ")[1].strip(),
                    "KL Divergence": float(lines[i+1].split(": ")[1].strip()),
                    "Latents": f"GAN -->  {model}",
                })

    df = pd.DataFrame(data)
    print(df)

    s = 3
    plt.figure(figsize=(6, s))
    sns.heatmap(df.pivot("Latents", "Attribute", "KL Divergence"), annot=True, fmt='.2f', cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("KL Divergence")
    plt.tight_layout()
    plt.savefig(f"plots/kl_divergence_cifar.png", dpi=300)

def parse_nf_training_accuracies():
    models = ["001001", "006001", "011001", "016001", "021001", "026001"]
    attributes = ["Bald", "Blond_Hair", "Wavy_Hair", "Smiling", "Eyeglasses", 
                  "Heavy_Makeup", "Male", "No_Beard", "Pale_Skin", "Young"]
    
    accuracies = {attr : [] for attr in attributes}
    
    for model in models:
        with open (f"probe_log_{model}_celeba.txt", "r") as file:
            lines = file.readlines()

        for i in range(0, len(lines), 4):
            attr = lines[i].split(" ")[0].strip()
            print(attr)
            print(model)
            accuracies[attr].append(float(lines[i+3].split(": ")[1].strip()[:-1]))
    
    plt.figure(figsize=(4, 1.5))
    for attr in attributes:
        plt.plot([1, 6, 11, 16, 21, 26], accuracies[attr], label=attr)
    
    # create vertical line at x = 6
    plt.axvline(x=6, color='grey', linestyle='--', linewidth=0.5)
    
    plt.legend(loc="lower right", ncol=2)
    plt.xlabel("NF Training Epoch")
    plt.ylabel("Probe Accuracy")
    # plt.title("Probe Accuracy on CelebA NF Latent Space Frozen at Various Epochs")
    plt.savefig("plots/nf_training_accuracies.png", dpi=400)
    # plt.show()
    

def parse_post_map_results(models, models2=[], with_gan=False, vae_training=False):
    global accuracies, per_attr_acc, data

    if vae_training:
        data = []

    if not models2:
        models2 = models
    
    atrributes_to_keep = ["Bald", "Blond_Hair", "Wavy_Hair", "Smiling", "Eyeglasses", 
                          "Heavy_Makeup", "Male", "No_Beard", "Pale_Skin", "Young"]

    if not with_gan:
        for model1 in models:
            for model2 in models2:
                if model1 != model2:
                    if os.path.isfile(f"post_map_probe_logs_{model1}_{model2}.txt"):
                        with open (f"post_map_probe_logs_{model1}_{model2}.txt", "r") as file:
                            lines = file.readlines()

                        for i in range(0, len(lines), 7):
                            attribute = lines[i].split(": ")[1].strip()
                            accuracies[model2].append(float(lines[i+3].split(": ")[1].strip()))
                            if len(accuracies[model2]) < 40:
                                per_attr_acc.append({"Model": model2, "Attribute": attribute, "Accuracy": float(lines[i+3].split(": ")[1].strip()) * 100})

                            if attribute not in atrributes_to_keep:
                                continue
                            data.append({
                                "Attribute": lines[i].split(": ")[1].strip(),
                                "Accuracy": float(lines[i+3].split(": ")[1].strip()),
                                "Percent Delta Accuracy": int(float(lines[i+5].split(": ")[1].strip())), 
                                # "MSE": float(lines[i+6].split(": ")[1].strip()),
                                # "Cosine Similarity": float(lines[i+7].split(": ")[1].strip()),
                                "Percentage Matches": int(float(lines[i+6].split(": ")[1].strip()) * 100),
                                "Map": f"{model_to_name[model1]} $\\rightarrow$ {model_to_name[model2]}"
                            })

                        
    else:
        for model in models:
            with open (f"post_map_probe_logs_gan_{model}.txt", "r") as file:
                lines = file.readlines()
                for i in range(0, len(lines), 2):
                    attribute = lines[i].split(": ")[1].strip()
                    if attribute not in atrributes_to_keep:
                        continue

                    data.append({
                        "Attribute": lines[i].split(": ")[1].strip(),
                        "Percentage Matches": int(float(lines[i+1].split(": ")[1].strip()) * 100),
                        "Map": f"GAN $\\rightarrow$ {model_to_name[model]}"
                    })

    df = pd.DataFrame(data)
    print(df)

    df2 = pd.DataFrame(per_attr_acc)
    print(df2)

    if with_gan:
        plt.figure(figsize=(3.5, 3.5))
        a = sns.heatmap(df.pivot(index="Map", columns="Attribute", values="Percentage Matches"), annot=True, fmt=".0f", cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
        a.set_xticklabels(a.get_xmajorticklabels(), fontsize = 8)
        a.set_yticklabels(a.get_ymajorticklabels(), fontsize = 8)
        a.set_xlabel("Attribute", fontsize=8)
        a.set_ylabel("Map", fontsize=8)
        plt.savefig(f"plots/percentage_matches_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)
    else:
        plt.figure(figsize=(3.5, 3))
        a = sns.heatmap(df.pivot(index="Map", columns="Attribute", values="Percent Delta Accuracy"), annot=True, vmin=-43, vmax=43, cmap=sns.diverging_palette(20, 145, s=60, as_cmap=True), cbar=False)
        a.set_xticklabels(a.get_xmajorticklabels(), fontsize = 8)
        a.set_yticklabels(a.get_ymajorticklabels(), fontsize = 8)
        a.set_xlabel("Attribute", fontsize=8)
        a.set_ylabel("Map", fontsize=8)
        plt.savefig(f"plots/percent_delta_accuracy_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)


if __name__ == '__main__':
    models = ["vae-diffusion", "vqvae", "dm", "nf"]

    parse_post_map_results(models, models, False)
    parse_post_map_results(models, models, True)
    parse_training_accuracies(models)

    parse_nf_training_accuracies()

