import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img_ind = '014998' # 014995 # 001986

# encoders = ["vae-21", "vqvae", "nf-celeba"]
# decoders = ["gan", "vae-21", "vqvae", "nf-celeba"]
# dataset = 'celeba'

# encoders = ["ae1", "nf"]
# decoders = ["gan", "ae1", "nf"]
# dataset = 'cifar'

encoders = ['vae-epoch-0', 'vae-epoch-0-1', 'vae-epoch-1', 'vae-21']
decoders = ['self', 'vae-21', 'gan']
dataset = 'cifar'

fig, axs = plt.subplots(len(encoders), len(decoders), figsize=(len(decoders), len(encoders)))

def n(name):
    if name == 'vae-epoch-0':
        return 'vae 0'
    elif name == 'vae-epoch-0-1':
        return 'vae 0.1'
    elif name == 'vae-epoch-1':
        return 'vae 1'
    elif name == 'vae-21':
        return 'vae'
    elif name == 'nf-celeba':
        return 'nf'
    else:
        return name

latent_mses_celeba = { 
    "vae" : {"vae": 0.000000, "vqvae": 0.435918, "nf": 3.564455, "gan": None},
    "vqvae" : {"vae": 0.456590, "vqvae": 0.000000, "nf": 4.276381, "gan": None},
    "nf" : {"vae": 0.594922, "vqvae": 0.321784, "nf": 0.000000, "gan": None}
}

latent_mses_cifar = {
    "ae" : {"ae": 0.000000, "nf": 8.737329, "gan": None},
    "nf" : {"ae": 14.422283, "nf": 0.000000, "gan": None}
}

latent_mses_celeba_vae = {
    "vae 0" : {"vae": 1.32, 'self': 0.0, 'gan': None},
    "vae 0.1" : {"vae": 0.47, 'self': 0.0, 'gan': None},
    "vae 1" : {"vae": 0.22, 'self': 0.0, 'gan': None},
    "vae" : {"vae": 0.0, 'self': 0.0, 'gan': None},
}

# Load images
for i, e in enumerate(encoders):
    for j, d in enumerate(decoders):
        if e == d or d == 'self':
            img = mpimg.imread(f"mapped/{dataset}/{img_ind}_{e}.jpg")
        else:
            img = mpimg.imread(f"mapped/{dataset}/{img_ind}_{e}_to_{d}_{'w+' if d == 'gan' else 'z'}_linear.jpg")

        axs[i, j].imshow(img)
        
        d_ = 'ae' if d == 'ae1' else d
        e_ = 'ae' if e == 'ae1' else e

        if i == len(encoders) - 1:
            axs[i, j].set_xlabel(n(d_), fontsize=10)
        if j == 0:
            axs[i, j].set_ylabel(n(e_), fontsize=10)
        
        l = latent_mses_celeba_vae[n(e_)][n(d_)]
        axs[i, j].set_title('' if l is None else round(l, 2), fontsize=8)
        
        if d == 'self': # e == d
            for spine in axs[i, j].spines.values():
                spine.set_edgecolor('red')


for ax in axs.flatten():
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.tight_layout()
plt.savefig(f'plots/{dataset}_mapped_example_vae.png', dpi=300)
