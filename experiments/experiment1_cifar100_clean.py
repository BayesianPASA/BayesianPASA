#!/usr/bin/env python3
"""CIFAR-100 Clean Experiment - Activation Function Comparison"""

import argparse
import torch
from bayesian_pasa.activations import *
from bayesian_pasa.normalization import *
from bayesian_pasa.models import ResNet18_CIFAR100
from bayesian_pasa.utils import set_seed, get_cifar100_loaders, train_and_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    activations = ['relu', 'leakyrelu', 'gelu', 'swish', 'mish', 'pasa', 'bayesian_pasa']
    results = {}
    
    for act in activations:
        model = ResNet18_CIFAR100(activation=act).to(device)
        trainloader, testloader = get_cifar100_loaders(batch_size=128)
        best_acc, smoothed_acc = train_and_evaluate(
            model, trainloader, testloader, 
            epochs=args.epochs, lr=args.lr
        )
        results[act] = {'best': best_acc, 'smoothed': smoothed_acc}
    
    print_results(results)
    save_results(results, 'cifar100_results.csv')

if __name__ == '__main__':
    main()
