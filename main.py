import json
import pandas as pd
from utils.preproc import (
    transform_data,
    reverse_transformers,
    set_seed,
)
import torch
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

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    logging.info("Reading params...")
    logging.info("Cuda:", torch.cuda.is_available())

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

    mlflow.set_tracking_uri(uri=params["tracking_uri"])
    mlflow.set_experiment("syntikka_experiments")

    set_seed(params["seed"])
    dataset = pd.read_csv(params["data_path"], delimiter=params["delimiter"])

    hyperparams = {
        "epochs": 20,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "kl_anneal_epochs": 10,
        "kl_anneal_method": "stable",
        "starting_kl_weight": 1.0,
        "target_kl_weight": 2.0,
        "dropout": 0.1,
        "latent_dim": 16,
        "hidden_dim": 64,
        "activation_function": torch.nn.ReLU,
        "optimizer": torch.optim.Adam,
        "scaler": "standard",
        "num_layers": 2,
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
        pre_proc_method=hyperparams["scaler"],
    )

    x_train_split, x_val_split = train_test_split(
        x_train, test_size=0.2, random_state=params["seed"]
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train_split, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val_split, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    encoder = Encoder(
        input_dim=x_train.shape[1],
        latent_dim=hyperparams["latent_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
        activation=hyperparams["activation_function"],
        num_layers=hyperparams["num_layers"],
    )

    decoder = Decoder(
        latent_dim=hyperparams["latent_dim"],
        num_continuous=num_continuous,
        num_categories=num_categories,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
        activation=hyperparams["activation_function"],
        num_layers=hyperparams["num_layers"],
    )

    mlflow.start_run()
    mlflow.log_params(hyperparams)

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

    vae.fit(
        train_loader=train_loader, 
        val_loader=val_loader, 
        n_epochs=hyperparams["epochs"], 
        early_stopper=early_stopper
    )

    syn_t = vae.generate(len(dataset))

    syn_df_t = pd.DataFrame(
        syn_t.cpu().detach().numpy(), columns=reordered_cols)
    syn_df: pd.DataFrame = reverse_transformers(
        syn_df_t,
        dataset.columns,
        cont_transformers=cont_trans,
        cat_transformers=cat_trans,
        pre_proc_method=hyperparams["scaler"],
    )

    syn_df.to_csv(params["output_path"] + "synthetic.csv")

    mlflow.pytorch.log_model(
        vae,
        "model",
        signature=infer_signature(model_input=dataset, model_output=syn_df),
    )
    mlflow.end_run()
