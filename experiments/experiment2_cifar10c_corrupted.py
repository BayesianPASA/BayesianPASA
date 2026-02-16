#!/usr/bin/env python3
"""CIFAR-10-C Corrupted Experiment - All Activations + Bayesian R‑LayerNorm"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayesian_pasa.activations import *
from bayesian_pasa.normalization import *
from bayesian_pasa.models import EfficientCNN
from bayesian_pasa.utils import set_seed, CachedNoisyCIFAR10, ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import gc
import os


def train_and_evaluate(norm_type, activation, num_epochs=100, lr=0.001,
                       train_samples=None, test_samples=None, device='cuda'):
    """Train a model on mixed corruptions, evaluate on each noise type."""
    
    if train_samples is None:
        train_samples = 50000
    if test_samples is None:
        test_samples = 10000
    
    print(f"\n>>> Training: norm={norm_type}, act={activation}")
    
    model = EfficientCNN(norm_type=norm_type, activation=activation, num_classes=10).to(device)
    
    # Training: mixed corruptions
    noise_types = ['gaussian', 'shot_noise', 'blur', 'contrast']
    datasets = []
    for noise in noise_types:
        ds = CachedNoisyCIFAR10(noise_type=noise, severity=3,
                                 num_samples=train_samples//4, train=True)
        datasets.append(ds)
    train_loader = DataLoader(ConcatDataset(datasets), batch_size=64,
                              shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'acc': 100.*correct/total})
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            train_acc = 100. * correct / total
            print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%")
    
    # Evaluation
    model.eval()
    test_results = {}
    for noise in noise_types:
        test_ds = CachedNoisyCIFAR10(noise_type=noise, severity=3,
                                      num_samples=test_samples//4, train=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 num_workers=2, pin_memory=True)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        test_results[noise] = acc
        print(f"    {noise}: {acc:.2f}%")
    
    return test_results, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/cifar10c_results.csv')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define experiments
    activations = ['relu', 'leakyrelu', 'gelu', 'swish', 'mish', 'pasa', 'bayesian_pasa']
    experiments = []
    
    # LayerNorm baseline
    for act in activations:
        experiments.append(('layer', act, f"{act}+LayerNorm"))
    
    # Bayesian R-LayerNorm for key activations
    key_acts = ['relu', 'gelu', 'mish', 'bayesian_pasa']
    for act in key_acts:
        experiments.append(('bayesian_r_layer', act, f"{act}+B-RLN"))
    
    print(f"Total experiments: {len(experiments)}")
    all_results = {}
    weights_data = {}
    
    for norm, act, label in experiments:
        test_results, model = train_and_evaluate(
            norm_type=norm,
            activation=act,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        all_results[label] = test_results
        
        # Collect weights for PASA models
        if 'pasa' in act.lower():
            ds = CachedNoisyCIFAR10(noise_type='gaussian', severity=3,
                                     num_samples=500, train=False)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            model.eval()
            all_w = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    if isinstance(model.act3, (PASA, BayesianPASA)):
                        _, w = model(inputs, return_weights=True)
                        all_w.append(w.cpu())
            if all_w:
                weights_data[label] = torch.cat(all_w, dim=0)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    save_results(all_results, weights_data, args.output)
    print(f"\n✅ Results saved to {args.output}")


def save_results(all_results, weights_data, output_path):
    """Save results to CSV and generate plots"""
    noise_types = ['gaussian', 'shot_noise', 'blur', 'contrast']
    
    # Create DataFrame
    rows = []
    for label, results_dict in all_results.items():
        for noise in noise_types:
            rows.append({
                'Model': label,
                'Noise': noise,
                'Accuracy': results_dict[noise]
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    # Calculate averages
    avg_accs = {}
    for label in all_results.keys():
        avg_accs[label] = np.mean([all_results[label][n] for n in noise_types])
    
    # Print ranking
    sorted_results = sorted(avg_accs.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "="*80)
    print("📊 CIFAR-10-C Results Ranking")
    print("="*80)
    print(f"{'Rank':<4} {'Model':<40} {'Avg Acc':>10}")
    for rank, (label, acc) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {label:<40} {acc:>10.2f}%")
    
    return df


if __name__ == '__main__':
    main()
