import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # ... (implementation)

class RLayerNorm(nn.Module):
    def __init__(self, num_features, lambda_init=0.01, eps=1e-5):
        super().__init__()
        # ... (implementation)

class BayesianRLayerNorm(nn.Module):
    def __init__(self, num_features, lambda_init=0.01, eps=1e-5):
        super().__init__()
        # ... (implementation with psi function)
