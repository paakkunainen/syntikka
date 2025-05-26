<div align="center">

# syntikka

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](http://www.python.org)
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://neovim.io)

<img alt="syntikka" height="280" src="/assets/syntikka.png" />
</div>

Syntikka is a VAE (Variational Autoencoder) based synthetic data generation
toolkit used to generate synthetic heterogeneous tabular data that mimics the
statistical properties of the training data. It is a work in progress project
that was founded during working on my Master's thesis. It is not a production-proof
tool and should be used with caution especially when working with sensitive data.
Currently it lacks features to robustly prevent sensitive information leaking
to the synthetic data from the original training data.

## MLflow

The training process utilizes MLflow. In order to run the training,
start the MLflow tracking server first (if running locally).

```bash
mlflow server --host 127.0.0.1 --port 8080
```
