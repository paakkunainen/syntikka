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


def run_experiment(config: dict, model_index: int, dataset: pd.DataFrame, params: dict):
    model_name = f"M{model_index}"
    logging.info(f"Running {model_name}...")

    hyperparams = {
        "epochs": 5,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "kl_anneal_epochs": 10,
        "activation_function": torch.nn.ReLU,
        "optimizer": torch.optim.Adam,
        **config
    }

    mlflow.start_run(run_name=model_name)
    mlflow.set_tag("model_name", model_name)
    mlflow.log_params(hyperparams)

    set_seed(params["seed"])
    (
        x_train,
        _,
        reordered_cols,
        cont_trans,
        cat_trans,
        num_categories,
        num_continuous,
    ) = transform_data(
        original_data=dataset,
        cat_cols=[k for k, v in params["columns"].items() if v.get("sdtype") == "categorical"],
        cont_cols=[k for k, v in params["columns"].items() if v.get("sdtype") == "numerical"],
        pre_proc_method=hyperparams["scaler"],
    )

    x_train_split, x_temp_split = train_test_split(x_train, test_size=0.2, random_state=params["seed"])
    x_val_split, x_eval_split = train_test_split(x_temp_split, test_size=0.5, random_state=params["seed"])

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train_split, dtype=torch.float32)), batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val_split, dtype=torch.float32)), batch_size=hyperparams["batch_size"], shuffle=False)
    eval_df_t = pd.DataFrame(x_eval_split, columns=reordered_cols)

    encoder = Encoder(
        input_dim=x_train.shape[1],
        latent_dim=hyperparams["latent_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        dropout=0.1,
        activation=hyperparams["activation_function"],
        num_layers=hyperparams["num_layers"],
    )

    decoder = Decoder(
        latent_dim=hyperparams["latent_dim"],
        num_continuous=num_continuous,
        num_categories=num_categories,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=0.1,
        activation=hyperparams["activation_function"],
        num_layers=hyperparams["num_layers"],
    )

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        lr=hyperparams["learning_rate"],
        kl_anneal_epochs=hyperparams["kl_anneal_epochs"],
        starting_kl_weight=hyperparams["starting_kl_weight"],
        target_kl_weight=hyperparams["target_kl_weight"],
        optimizer=hyperparams["optimizer"],
        kl_anneal_method=hyperparams["kl_anneal_method"],
    )

    with open("model_summary.txt", "w") as f:
        f.write(str(summary(vae)))
    mlflow.log_artifact("model_summary.txt")

    early_stopper = EarlyStopping(patience=5, delta=0.0, verbose=True)

    vae.fit(train_loader=train_loader, 
            val_loader=val_loader, 
            n_epochs=hyperparams["epochs"], 
            early_stopper=early_stopper)

    syn_t = vae.generate(len(eval_df_t))
    syn_df_t = pd.DataFrame(syn_t.cpu().detach().numpy(), columns=reordered_cols)

    syn_df = reverse_transformers(
        syn_df_t,
        dataset.columns,
        cont_transformers=cont_trans,
        cat_transformers=cat_trans,
        pre_proc_method=hyperparams["scaler"],
    )
    
    eval_df = reverse_transformers(
        eval_df_t,
        dataset.columns,
        cont_transformers=cont_trans,
        cat_transformers=cat_trans,
        pre_proc_method=hyperparams["scaler"],
    )

    mlflow.pytorch.log_model(
        vae, "model",
        signature=infer_signature(model_input=dataset, model_output=syn_df),
    )

    real_cv = calculate_cramers_v_matrix(eval_df, [k for k, v in params["columns"].items() if v.get("sdtype") == "categorical"])
    synth_cv = calculate_cramers_v_matrix(syn_df, [k for k, v in params["columns"].items() if v.get("sdtype") == "categorical"])
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(real_cv, cmap="coolwarm", vmin=0, vmax=1, annot=True, fmt=".2f", square=True, ax=ax[0])
    ax[0].set_title("Real (Eval)")
    sns.heatmap(synth_cv, cmap="coolwarm", vmin=0, vmax=1, annot=True, fmt=".2f", square=True, ax=ax[1])
    ax[1].set_title("Synthetic")
    plt.tight_layout()
    plt.savefig("cramers_v_heatmap.png")
    mlflow.log_artifact("cramers_v_heatmap.png")
    plt.close()

    fig = compare_distributions(eval_df, syn_df,
        [k for k, v in params["columns"].items() if v.get("sdtype") == "categorical"],
        [k for k, v in params["columns"].items() if v.get("sdtype") == "numerical"]
    )
    fig.savefig("distribution_comparison.png")
    mlflow.log_artifact("distribution_comparison.png")
    plt.close(fig)

    mlflow.end_run()