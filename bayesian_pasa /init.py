"""
BayesianPASA: Probabilistic Adaptive Sigmoidal Activation with Uncertainty Quantification
"""

from .activations import PASA, BayesianPASA, Mish, Swish
from .normalization import StandardLayerNorm, RLayerNorm, BayesianRLayerNorm
from .models import EfficientCNN, ResNet18_CIFAR100
from .utils import set_seed, get_cIFAR100_loaders, get_cifar10c_loaders, train_and_evaluate

__version__ = "1.0.0"
__author__ = "Mohsen Mostafa"
__email__ = "mohsen.mostafa.ai@outlook.com"

__all__ = [
    'PASA',
    'BayesianPASA',
    'Mish',
    'Swish',
    'StandardLayerNorm',
    'RLayerNorm',
    'BayesianRLayerNorm',
    'EfficientCNN',
    'ResNet18_CIFAR100',
    'set_seed',
    'get_cifar100_loaders',
    'get_cifar10c_loaders',
    'train_and_evaluate'
]
