import json
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2_contingency
from sdmetrics.reports.single_table import QualityReport
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def plot_training_losses(
    elbo, recon_cat, recon_cont, kl, beta, *, log_scale=False, save_path=None
):
    sns_palette = plt.get_cmap("tab10")
    epochs = range(1, len(elbo) + 1)
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"hspace": 0.08}
    )

    ax[0].plot(epochs, elbo, label="ELBO", lw=2, color=sns_palette(0))
    ax[0].plot(epochs, kl, label="β·KL", lw=1.8, color=sns_palette(1), ls="--")
    ax[0].set_ylabel("Loss value", fontsize=12)
    ax[0].set_title("Optimisation progress", fontsize=14, pad=4)
    ax[0].grid(ls=":", alpha=0.3)
    ax[0].legend(fontsize=11, frameon=False)
    if log_scale:
        ax[0].set_yscale("log")

    ax[1].plot(epochs, recon_cat, label="Categorical", lw=1.8, color=sns_palette(2))
    ax[1].plot(epochs, recon_cont, label="Continuous", lw=1.8, color=sns_palette(3))
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Reconstruction", fontsize=12)
    ax[1].grid(ls=":", alpha=0.3)
    ax[1].legend(fontsize=11, frameon=False)
    if log_scale:
        ax[1].set_yscale("log")

    ax_beta = ax[0].twinx()
    ax_beta.plot(epochs, beta, color="grey", lw=1.0, alpha=0.5)
    ax_beta.set_ylabel(r"KL weight $\beta$", fontsize=10, color="grey")
    ax_beta.tick_params(axis="y", labelsize=9, colors="grey")
    ax_beta.set_ylim(0, 1.05)

    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.tick_params(axis="both", labelsize=10)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return (
        np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        if (kcorr - 1) > 0 and (rcorr - 1) > 0
        else np.nan
    )


def calculate_cramers_v_matrix(df, categorical_columns):
    n = len(categorical_columns)
    cramers_v_matrix = pd.DataFrame(
        np.zeros((n, n)), index=categorical_columns, columns=categorical_columns
    )

    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 != col2:
                if df[col1].nunique() > 1 and df[col2].nunique() > 1:
                    confusion_matrix = pd.crosstab(df[col1], df[col2])
                    cramers_v_matrix.loc[col1, col2] = cramers_v(confusion_matrix)
                else:
                    cramers_v_matrix.loc[col1, col2] = np.nan
            else:
                cramers_v_matrix.loc[col1, col2] = 1.0

    return cramers_v_matrix


def compare_distributions(
    real_df, synthetic_df, categorical_columns, continuous_columns
):
    num_plots = len(categorical_columns) + len(continuous_columns)
    num_cols = 4
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    plot_index = 0

    for col in categorical_columns:
        ax = axes[plot_index]
        if col in ["kunta", "icpc2"]:
            real_counts = real_df[col].value_counts().nlargest(50)
            synthetic_counts = (
                synthetic_df[col].value_counts().reindex(real_counts.index)
            )
            sns.barplot(
                x=real_counts.index,
                y=real_counts.values,
                ax=ax,
                color="darkblue",
                alpha=0.8,
                label="Real",
                edgecolor="darkblue",
            )
            sns.barplot(
                x=synthetic_counts.index,
                y=synthetic_counts.values,
                ax=ax,
                color="cyan",
                alpha=0.5,
                label="Synthetic",
                edgecolor="cyan",
            )
            ax.set_title(col.capitalize())
            ax.tick_params(axis="x", rotation=90, labelsize=4)
            ax.tick_params(axis="y", labelsize=4)
        else:
            sns.countplot(
                x=col,
                data=real_df,
                ax=ax,
                order=real_df[col].value_counts().index,
                color="darkblue",
                alpha=0.8,
                label="Real",
                edgecolor="darkblue",
            )
            sns.countplot(
                x=col,
                data=synthetic_df,
                ax=ax,
                order=real_df[col].value_counts().index,
                color="cyan",
                alpha=0.5,
                label="Synthetic",
                edgecolor="cyan",
            )
            ax.set_title(f"{col.capitalize()}")
            ax.tick_params(axis="x", rotation=90, labelsize=4)
            ax.tick_params(axis="y", labelsize=4)
        ax.legend()
        ax.set(xlabel=None)
        plot_index += 1

    for col in continuous_columns:
        ax = axes[plot_index]
        sns.histplot(
            real_df[col],
            kde=True,
            ax=ax,
            color="darkblue",
            label="Real",
            alpha=0.8,
            edgecolor="darkblue",
        )
        sns.histplot(
            synthetic_df[col],
            kde=True,
            ax=ax,
            color="cyan",
            label="Synthetic",
            alpha=0.5,
            edgecolor="cyan",
        )
        ax.set_title(col.capitalize())
        ax.tick_params(axis="x", rotation=90, labelsize=4)
        ax.tick_params(axis="y", labelsize=4)
        ax.set(xlabel=None)
        ax.legend()
        plot_index += 1

    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig


def plot_latent_space(
    vae,
    dataloader,
    label_series,
    method: str = "TSNE",
    max_points: int = 5000,
    title: str = "Latent Space",
    figsize=(6, 6),
    cmap="tab10",
):
    vae.eval()
    zs, ys = [], []

    with torch.no_grad():
        ptr = 0
        for (batch,) in dataloader:
            batch = batch.to(vae.device)
            mu, _ = vae.encoder(batch)
            zs.append(mu.cpu().numpy())
            ys.append(label_series.iloc[ptr : ptr + len(batch)].values)
            ptr += len(batch)

    z_mat = np.concatenate(zs)
    y_vec = np.concatenate(ys)

    if len(z_mat) > max_points:
        idx = np.random.choice(len(z_mat), max_points, replace=False)
        z_mat, y_vec = z_mat[idx], y_vec[idx]

    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        reducer = TSNE(n_components=2, perplexity=40, learning_rate="auto")
    else:
        import umap

        reducer = umap.UMAP(n_components=2)

    z_2d = reducer.fit_transform(z_mat)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_vec)

    plt.figure(figsize=figsize)
    sns.scatterplot(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        hue=y_enc,
        palette=cmap,
        s=15,
        linewidth=0,
        alpha=0.8,
        legend=False,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(ls=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def distribution_metrics(real_data, synthetic_data, params_path):
    report = QualityReport()
    with open(params_path) as f:
        data = f.read()
    metadata = json.loads(data)
    report.generate(real_data, synthetic_data, metadata)
    return report
