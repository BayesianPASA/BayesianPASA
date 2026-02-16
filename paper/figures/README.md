# Figures for BayesianPASA Paper

This directory contains all figures used in the BayesianPASA paper.

## Figure Files

| Filename | Description |
|----------|-------------|
| `cifar100_results.png` | CIFAR-100 activation comparison bar chart |
| `cifar10c_comparison.png` | CIFAR-10-C top 10 configurations comparison |
| `pasa_weights_distribution.png` | Softmax weight distributions for PASA models |
| `improvement_over_baseline.png` | Improvement over ReLU baseline on CIFAR-10-C |
| `cifar100_training_curves.png` | Training curves for CIFAR-100 experiment |

## Regenerating Figures

To regenerate all figures:

```bash
python experiments/generate_plots.py --all
