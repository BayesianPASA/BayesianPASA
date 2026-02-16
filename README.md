# BayesianPASA: Probabilistic Adaptive Sigmoidal Activation with Uncertainty Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📊 State-of-the-Art Results

| Model | CIFAR-100 (Clean) | CIFAR-10-C (Avg) | Improvement |
|-------|-------------------|------------------|-------------|
| **BayesianPASA + B-RLN** | **76.38%** | **53.91%** | **+1.87%** 🏆 |
| ReLU + LayerNorm | 75.68% | 52.04% | baseline |
| GELU + LayerNorm | 75.98% | 51.00% | - |

## 🔬 Overview

BayesianPASA extends the original PASA with rigorous probabilistic foundations and uncertainty quantification. It integrates:

- **ψ‑function stabilization** from Bayesian R‑LayerNorm
- **Variational evidence scores** for adaptive mixing
- **Learnable temperature** for softmax control
- **Provable Lipschitz continuity and gradient stability**

## 🚀 Quick Start

```python
from bayesian_pasa.activations import BayesianPASA
from bayesian_pasa.normalization import BayesianRLayerNorm

# Use in your model
activation = BayesianPASA()
norm = BayesianRLayerNorm(num_features=64)
