import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tueplots import bundles

plt.rcParams.update(bundles.icml2024())
from run import test_exact_latent_space_mapping

encoders = ["gan", "random", "vae-diffusion", "vqvae", "nf", "dm"]
decoders = ["gan"]
dataset = 'celeba'

fig, axs = plt.subplots(len(decoders), len(encoders), figsize=(len(encoders), len(decoders) + 0.5))

def n(name):
    if name == 'random':
        return 'Random'
    elif name == 'vae-diffusion':
        return 'VAE'
    else:
        return name.upper()

ind = 9060
# Load images
for i, e in enumerate(encoders):
    for j, d in enumerate(decoders):
        if e == d or d == 'self':
            img = mpimg.imread(f"data/{dataset}_gen/{str(ind).zfill(4 if dataset == 'cifar' else 6)}.jpg")
        else:
            img = mpimg.imread(f"mapped/{dataset}/{str(ind).zfill(6)}{'_gen' if decoders == ['gan'] else ''}_{e}_to_{d}_linear.jpg")

        axs[i].imshow(img)
        axs[i].set_xlabel(n(e), fontsize=14)

        mse = 0
        if e != 'gan':
            mse = test_exact_latent_space_mapping(e, d, False, True, False, e in ["nf", "dm"], ind)
        
        axs[i].set_title(round(mse, 2), fontsize=16, color='gray')

for ax in axs.flatten():
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.savefig(f'plots/{dataset}_mapped_example_{str(ind).zfill(6)}_gen.png', dpi=300)
