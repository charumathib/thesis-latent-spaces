import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from tueplots import figsizes, fonts, bundles

plt.rcParams.update(bundles.icml2024())
from run import test_exact_latent_space_mapping, Model

# celeba: 9086, 9067, 9061, 9018
ind = 9018

# cifar 1910, 403, 905, 3908 (a failure case)
# cifar_range = []
# for i in range(10):
#     cifar_range += list(range((500 * i) + 401, (500 * i) + 411))

# encoders = ["vae-21", "vqvae", "nf-celeba"]
# decoders = ["gan", "vae-21", "vqvae", "nf-celeba"]
# dataset = 'celeba'

# encoders = ['vae-epoch-0', 'vae-epoch-0-1', 'vae-epoch-1', 'vae-21']
# decoders = ['self', 'vae-21', 'gan']

# encoders = ['ae1', 'ae2']
# decoders = ['ae1', 'ae2']
# dataset = 'cifar'

encoders = ["random", "vae-diffusion", "vqvae", "nf", "dm"]
decoders = ["vae-diffusion", "vqvae", "nf", "dm"]
dataset = 'celeba'

# encoders = ["vae-diffusion", "nf", "dm"]
# decoders = ["gan", "vae-diffusion", "nf", "dm"]
# dataset = 'cifar'

# encoders = ["gan", "vae-diffusion", "nf", "dm"]
# decoders = ["gan"]
# dataset = 'cifar'

fig, axs = plt.subplots(len(decoders), len(encoders), figsize=(len(encoders), len(decoders) + 0.5))

def n(name):
    if name == 'random':
        return 'Random'
    if name == 'vae-diffusion':
        return 'VAE'
    else:
        return name.upper()

# Load images
for j, e in enumerate(encoders):
    for i, d in enumerate(decoders):
        if e == d or d == 'self' or (e == 'vae-diffusion' and d == 'vae'):
            if dataset == 'cifar':
                img = mpimg.imread(f"mapped/{dataset}/{str(cifar_range.index(ind)).zfill(6)}_{e}.jpg")
            else:
                img = mpimg.imread(f"mapped/{dataset}/{str(ind).zfill(6)}{'_gen' if decoders == ['gan'] else ''}_{'vae' if d == 'vae-diffusion' else d}.jpg")
            # img = mpimg.imread(f"data/cifar_gen/{str(ind).zfill(4)}.jpg")
        else:
            img = mpimg.imread(f"mapped/{dataset}/{str(ind).zfill(6)}{'_gen' if decoders == ['gan'] else ''}_{e}_to_{'vae' if d == 'vae-diffusion' else d}_linear.jpg")

        axs[i, j].imshow(img)

        # start writing paper!!; make sure figures are consistent
        if i == len(decoders) - 1:
            axs[i, j].set_xlabel(n(e), fontsize=16)
        if j == 0:
            axs[i, j].set_ylabel(n(d), fontsize=16)


        mse = 0
        if e != 'gan' and e != d:
            mse = test_exact_latent_space_mapping(
                from_name=e, 
                to_=Model(d, dataset), 
                on_train_set=False, 
                synthetic=False, 
                save_images=False, 
                regularized=e in ['nf', 'dm'],
                ind=ind
            )
        
        axs[i, j].set_title(round(mse, 2) if mse is not None else '', color='gray', fontsize=14, pad=-0.5 if i != len(decoders) - 1 else 3)
        

        if e == d or (e == 'vae-diffusion' and d == 'vae'): # e == d
            for spine in axs[i, j].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            for spine in axs[i, j].spines.values():
                spine.set_linewidth(0)

fig.supxlabel("Encoder", fontsize=14)
fig.supylabel("Decoder", fontsize=14)

for ax in axs.flatten():
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.tight_layout()
plt.savefig(f'plots/{dataset}_mapped_example_{str(ind).zfill(6)}.png', dpi=300)
