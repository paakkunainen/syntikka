import json
import pandas as pd
from utils.preproc import (
    transform_data,
    reverse_transformers,
    set_seed,
)
from utils.plot import calculate_cramers_v_matrix, compare_distributions
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from core.decoder import Decoder
from core.encoder import Encoder
from core.vae import VAE, EarlyStopping
import mlflow
from mlflow.models.signature import infer_signature
import logging
import sys
import itertools


with open("params.json", "r") as f:
    params = json.load(f)

dataset = pd.read_csv(params["data_path"], delimiter=params["delimiter"])

kl_methods = ['standard', 'beta']
latent_dims = [8, 32]
num_layers = [1, 3]
scalers = ['standard', 'gmm']
hidden_dims = [64]

kl_weights = {
    'standard': {'starting_kl_weight': 1.0, 'target_kl_weight': 1.0},
    'beta': {'starting_kl_weight': 0.0, 'target_kl_weight': 4.0}
}

i = 1
for kl, latent, layers, scaler, hidden in itertools.product(kl_methods, latent_dims, num_layers, scalers, hidden_dims):
    config = {
        'latent_dim': latent,
        'num_layers': layers,
        'scaler': scaler,
        'hidden_dim': hidden,
        'kl_anneal_method': 'stable',
        'starting_kl_weight': kl_weights[kl]['starting_kl_weight'],
        'target_kl_weight': kl_weights[kl]['target_kl_weight'],
    }

    try:
        run_experiment(config=config, model_index=i, dataset=dataset, params=params)
    except Exception as e:
        import traceback
        logging.error(f"Experiment M{i} failed with error: {e}")
        traceback.print_exc()
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()

    i += 1
