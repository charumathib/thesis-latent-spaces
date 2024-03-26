import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
# with open("post_map_probe_logs_gan_vae_reencode.txt", "r") as file:
#     lines = file.readlines()

accuracies = {"vae": [], "vqvae": [],  "nf": []}
# per_attr_acc = {"vae": {}, "vqvae": {},  "nf": {}}
per_attr_acc = []
data = []


def parse_training_accuracies(models):
    data = []
    for model in models:
        with open (f"probe_logs_{model}.txt", "r") as file:
            lines = file.readlines()
        
        if model == 'vae':
            model = "vae (21 epochs)"
        if model == 'vae-epoch-1':
            model = "vae (1 epoch)"
        elif model == 'vae-epoch-0-1':
            model = "vae (0.1 epochs)"
        elif model == 'vae-epoch-0':
            model = "vae (0 epochs)"

        for i in range(0, len(lines), 6):
            print(lines[i+5].split(": ")[1].strip()[:-1])
            data.append({
                "Attribute": lines[i].split(" ")[0].strip(),
                "Latents": model,
                "Test Accuracy": float(lines[i+5].split(": ")[1].strip()[:-1])
            })
    
    df = pd.DataFrame(data)
    print(df)
    plt.figure(figsize=(20, 4))
    sns.heatmap(df.pivot("Latents", "Attribute", "Test Accuracy"), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Test Set Probe Accuracy")
    plt.tight_layout()
    plt.savefig(f"plots/test_accuracy_heatmap_raining.png", dpi=300)

    # plt.savefig(f"plots/percent_delta_accuracy_heatmap.png", dpi=300)
    
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
                    "Latents": f"gan -->  {model}",
                })

    df = pd.DataFrame(data)
    print(df)

    s = 3
    plt.figure(figsize=(6, s))
    sns.heatmap(df.pivot("Latents", "Attribute", "KL Divergence"), annot=True, fmt='.2f', cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("KL Divergence")
    plt.tight_layout()
    plt.savefig(f"plots/kl_divergence_cifar.png", dpi=300)

def parse_post_map_results(models, models2=[], with_gan=False, vae_training=False):
    global accuracies, per_attr_acc, data

    if vae_training:
        data = []

    if not models2:
        models2 = models
    
    atrributes_to_keep = [
        "Bangs", 
        "Black_Hair", 
        "Blond_Hair", 
        "Eyeglasses", 
        "Heavy_Makeup", 
        "Male", 
        "Mouth_Slightly_Open",
        "No_Beard",
        "Pale_Skin",
        "Smiling"
    ]


    if not with_gan:
        for model1 in models:
            for model2 in models2:
                if model1 != model2:
                    with open (f"post_map_probe_logs_{model1}_{model2}.txt", "r") as file:
                        lines = file.readlines()

                    for i in range(0, len(lines), 9):
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
                            "MSE": float(lines[i+6].split(": ")[1].strip()),
                            "Cosine Similarity": float(lines[i+7].split(": ")[1].strip()),
                            "Percentage Matches": int(float(lines[i+8].split(": ")[1].strip()) * 100),
                            "Latents": f"{model1} --> {model2}"
                        })

                        
    else:
        for model in models:
            with open (f"post_map_probe_logs_gan_{model}_gen.txt", "r") as file:
                lines = file.readlines()
                for i in range(0, len(lines), 4):
                    attribute = lines[i].split(": ")[1].strip()
                    if attribute not in atrributes_to_keep:
                        continue

                    data.append({
                        "Attribute": lines[i].split(": ")[1].strip(),
                        "MSE": float(lines[i+1].split(": ")[1].strip()), 
                        "Cosine Similarity": float(lines[i+2].split(": ")[1].strip()), 
                        "Percentage Matches": int(float(lines[i+3].split(": ")[1].strip())),
                        "Latents": f"gan -->  {model}",
                        "Accuracy": accuracies[model][i // 4]
                    })

    df = pd.DataFrame(data)
    print(df)

    df2 = pd.DataFrame(per_attr_acc)
    print(df2)



    if accuracies["vae"]:
        # Define a mask for values > 0.7
        mask = np.zeros_like(df.pivot("Latents", "Attribute", "MSE"))
        mask[df.pivot("Latents", "Attribute", "Accuracy") <= 0.7] = True
    else:
        mask = None

    
    s = 4 if vae_training else 5
    if with_gan or vae_training:
        # Create a heatmap for the MSE
        plt.figure(figsize=(10, s))
        sns.heatmap(df.pivot("Latents", "Attribute", "MSE"), annot=True, fmt=".2f", cmap=sns.cubehelix_palette(as_cmap=True), mask=mask)
        plt.title("MSE Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/mse_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)

        plt.figure(figsize=(10, s))
        sns.heatmap(df.pivot("Latents", "Attribute", "Cosine Similarity"), annot=True, fmt=".2f", cmap=sns.cubehelix_palette(as_cmap=True), mask=mask)
        plt.title("Cosine Similarity Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/cosine_similarity_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)

        plt.figure(figsize=(10, s))
        sns.heatmap(df.pivot("Latents", "Attribute", "Percentage Matches"), annot=True, fmt=".0f", cmap=sns.cubehelix_palette(as_cmap=True), mask=mask)
        plt.title("Percentage Matches Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/percentage_matches_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)

    if not with_gan:
        plt.figure(figsize=(16, 4))
        sns.heatmap(df2.pivot("Model", "Attribute", "Accuracy"), annot=True, fmt=".0f", cmap=sns.cubehelix_palette(as_cmap=True))
        plt.title("Test Set Probe Accuracy")
        plt.tight_layout()
        plt.savefig(f"plots/test_set_probe_accuracy_all.png", dpi=300)

        plt.figure(figsize=(10, 4))
        sns.heatmap(df.pivot("Latents", "Attribute", "Percent Delta Accuracy"), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), mask=mask)
        plt.title("Percent Delta Accuracy Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/percent_delta_accuracy_heatmap{'_vae' if vae_training else ''}_partial.png", dpi=300)


if __name__ == '__main__':
    models = ['ae', 'nf']
    parse_post_map_results_cifar(models)