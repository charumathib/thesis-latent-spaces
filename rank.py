import numpy as np
import pandas as pd
from joblib import load


def map_svd(from_, to_, layer, dataset):
    map = load(f"pickles/latent_mapping_{from_}_{to_}_{dataset}_{layer}_linear.joblib")
    weights = map.coef_

    singular_values = np.linalg.svd(weights, full_matrices=False, compute_uv=False)
    variance_explained = singular_values / np.sum(singular_values)
    cumulative_variance_explained = np.cumsum(variance_explained)

    return np.argmax(cumulative_variance_explained > 0.9)
    
# models = ['vae-21', 'vqvae', 'gan', 'nf-celeba']
models = ['ae1', 'nf', 'gan']

results = []
for model1 in models:
    for model2 in models:
        if model1 != model2 and model1 != 'gan':
            rank = map_svd(model1, model2, 'w+' if model2 == 'gan' else 'z', 'cifar')
            results.append({
                "Encoder Latent Space": model1,
                "Decoder Latent Space": model2,
                "Map Rank": rank
            })

df = pd.DataFrame(results)
print(df.to_latex(index=False))