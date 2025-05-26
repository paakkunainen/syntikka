import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import mlflow
from typing import List, Dict
from core.decoder import Decoder
from core.encoder import Encoder
from core.noiser import Noiser


class VAE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        lr: float = 1e-3,
        kl_anneal_epochs: int = 20,
        starting_kl_weight: float = 0.0,
        target_kl_weight: float = 1.0,
        optimizer: torch.optim.Optimizer = None,
        kl_anneal_method: str = "logistic",
    ):
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.decoder = decoder.to(decoder.device)
        self.noiser = Noiser(self.decoder.num_continuous).to(decoder.device)

        self.num_categories = decoder.num_categories
        self.num_continuous = decoder.num_continuous
        self.device = encoder.device

        self.lr = lr

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif callable(optimizer):
            self.optimizer = optimizer(self.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=3
        )

        self.kl_anneal_epochs = kl_anneal_epochs
        self.starting_kl_weight = starting_kl_weight
        self.target_kl_weight = target_kl_weight
        self.kl_anneal_method = kl_anneal_method

        self.history: Dict[str, List[float]] = {
            k: [] for k in ["elbo", "recon_cat", "recon_cont", "kl", "beta"]
        }

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        if torch.cuda.is_available():
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def generate(self, n: int, gumbel_temp: float = 0.67) -> torch.Tensor:
        z = torch.randn(n, self.encoder.latent_dim, device=self.device)
        out = self.decoder(z)

        # Process categoricals with Gumbel‑softmax
        processed = torch.empty_like(out)
        i = 0
        for n_cat in self.num_categories:
            processed[:, i : i + n_cat] = self.gumbel_softmax_sample(
                out[:, i : i + n_cat], gumbel_temp
            )
            i += n_cat

        # Continuous
        processed[:, -self.num_continuous :] = out[
            :, -self.num_continuous :
        ] + torch.exp(self.noiser(out[:, -self.num_continuous :])) * torch.randn_like(
            out[:, -self.num_continuous :]
        )
        return processed

    def _elbo_components(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)

        # categorical log‑likelihood
        cat_loglik, idx, i = 0.0, 0, 0
        for n_cat in self.num_categories:
            idx = torch.argmax(x[:, i : i + n_cat], dim=1)
            cat_loglik += -F.cross_entropy(
                recon[:, i : i + n_cat], idx, reduction="sum"
            )
            i += n_cat
        recon_cat = -cat_loglik

        # continuous log‑likelihood (gaussian)
        cont_mu = recon[:, -self.num_continuous :]
        cont_sigma = torch.exp(self.noiser(cont_mu))
        gauss = Normal(cont_mu, cont_sigma)
        cont_loglik = gauss.log_prob(x[:, -self.num_continuous :]).sum()
        recon_cont = -cont_loglik

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl.mean()
        return kl, recon_cat, recon_cont

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        n_epochs: int = 100,
        log_every: int = 1,
    ) -> None:
        for epoch in range(n_epochs):
            total_loss = 0.0
            total_kl = 0.0
            total_rc_cat = 0.0
            total_rc_cont = 0.0
            beta = self._get_kl_weight(epoch, method=self.kl_anneal_method)

            for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
                batch = batch.to(self.device)
                kl, rc_cat, rc_cont = self._elbo_components(batch)
                recon = rc_cat + rc_cont
                loss = recon + beta * kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl.item()
                total_rc_cat += rc_cat.item()
                total_rc_cont += rc_cont.item()

            avg_loss = total_loss / len(train_loader)
            avg_kl = total_kl / len(train_loader)
            avg_rc_cat = total_rc_cat / len(train_loader)
            avg_rc_cont = total_rc_cont / len(train_loader)

            self.history["elbo"].append(avg_loss)
            self.history["recon_cat"].append(avg_rc_cat)
            self.history["recon_cont"].append(avg_rc_cont)
            self.history["kl"].append(avg_kl)
            self.history["beta"].append(beta)

            # Validation step and scheduler update
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for (batch,) in val_loader:
                        batch = batch.to(self.device)
                        kl, rc_cat, rc_cont = self._elbo_components(batch)
                        val_loss += (rc_cat + rc_cont + beta * kl).item()
                    avg_val_loss = val_loss / len(val_loader)
                    self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step(avg_loss)

            mlflow.log_metrics(
                {
                    "elbo": avg_loss,
                    "val_elbo": avg_val_loss,
                    "recon_cat": avg_rc_cat,
                    "recon_cont": avg_rc_cont,
                    "kl": avg_kl,
                    "beta": beta,
                },
                step=epoch,
            )
            if epoch % log_every == 0:
                print(
                    f"[epoch {epoch}] β={beta:.3f} | nll={
                        avg_rc_cat + avg_rc_cont:.2f
                    } | kl_divergence={avg_kl:.2f} | elbo={avg_loss:.2f} | val loss = {
                        avg_val_loss:.2f
                    }"
                )

    def _get_kl_weight(self, epoch: int, method: str = "linear") -> float:
        if method == "linear":
            if epoch >= self.kl_anneal_epochs:
                return self.target_kl_weight
            slope = (
                self.target_kl_weight - self.starting_kl_weight
            ) / self.kl_anneal_epochs
            beta = self.starting_kl_weight + slope * epoch
            return beta
        if method == "logistic":
            progress = epoch / self.kl_anneal_epochs
            beta = float(self.target_kl_weight) / (1 + np.exp(-10 * (progress - 0.5)))
            return beta
        if method == "stable":
            return self.target_kl_weight
