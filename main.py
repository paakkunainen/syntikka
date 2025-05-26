import json
import pandas as pd
from utils.preproc import transform_data, reverse_transformers, unix_to_datetime
import torch
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from core.decoder import Decoder
from core.encoder import Encoder
from core.vae import VAE
import mlflow
from mlflow.models.signature import infer_signature
import logging

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("Syntikka")
    logging.info("Reading params...")
    with open("./params.json", "r") as params_json:
        params = json.load(params_json)
        categorical_columns = [
            key
            for key, value in params["columns"].items()
            if value.get("sdtype") == "categorical"
        ]
        continuous_columns = [
            key
            for key, value in params["columns"].items()
            if value.get("sdtype") == "numerical"
        ]

    dataset = pd.read_csv("./data/bank-full.csv", delimiter=";")

    hyperparams = {
        "epochs": 25,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "kl_anneal_epochs": 10,
        "kl_anneal_method": "logistic",
        "starting_kl_weight": 0.0,
        "target_kl_weight": 1.0,
        "dropout": 0.0,
        "latent_dim": 16,
        "hidden_dim": 64,
        "activation_function": torch.nn.Tanh,
        "optimizer": torch.optim.Adam,
        "scaler": "GMM",
        "num_layers": 3,
    }

    logging.info("Preprocessing data...")

    (
        x_train,
        _metric_set,
        reordered_cols,
        cont_trans,
        cat_trans,
        num_categories,
        num_continuous,
    ) = transform_data(
        original_data=dataset,
        cat_cols=categorical_columns,
        cont_cols=continuous_columns,
        pre_proc_method=params["scaler"],
    )

    x_train_split, x_val_split = train_test_split(
        x_train, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(x_train_split, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val_split, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    encoder = Encoder(
        input_dim=x_train.shape[1],
        latent_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
        activation=params["activation_function"],
        num_layers=params["num_layers"],
    )

    decoder = Decoder(
        latent_dim=params["latent_dim"],
        num_continuous=num_continuous,
        num_categories=num_categories,
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
        activation=params["activation_function"],
        num_layers=params["num_layers"],
    )

    mlflow.start_run()
    mlflow.log_params(params)

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        lr=params["learning_rate"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        starting_kl_weight=params["starting_kl_weight"],
        target_kl_weight=params["target_kl_weight"],
        optimizer=params["optimizer"],
        kl_anneal_method=params["kl_anneal_method"],
    )

    with open("model_summary.txt", "w") as f:
        f.write(str(summary(vae)))
    mlflow.log_artifact("model_summary.txt")

    vae.fit(train_loader=train_loader, val_loader=val_loader, n_epochs=params[" pochs"])

    syn_t = vae.generate(len(dataset))

    syn_df_t = pd.DataFrame(syn_t.cpu().detach().numpy(), columns=reordered_cols)
    syn_df: pd.DataFrame = reverse_transformers(
        syn_df_t,
        dataset.columns,
        cont_transformers=cont_trans,
        cat_transformers=cat_trans,
        pre_proc_method="GMM",
    )

    syn_df.to_csv("./data/synthetic.csv")

    mlflow.pytorch.log_model(
        vae,
        "model",
        signature=infer_signature(model_input=dataset, model_output=syn_df),
    )
    mlflow.end_run()
