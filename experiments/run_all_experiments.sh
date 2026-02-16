#!/bin/bash

# BayesianPASA Experiment Runner
# This script runs all experiments sequentially

echo "========================================="
echo "BayesianPASA: Running All Experiments"
echo "========================================="

# Set CUDA device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Create results directory
mkdir -p results/figures

# Experiment 1: CIFAR-100 Clean
echo ""
echo "[1/2] Running CIFAR-100 Clean Experiment..."
python experiments/experiment1_cifar100_clean.py \
    --epochs 50 \
    --lr 0.1 \
    --seed 42 \
    --output results/cifar100_results.csv

# Experiment 2: CIFAR-10-C Corrupted
echo ""
echo "[2/2] Running CIFAR-10-C Corrupted Experiment..."
python experiments/experiment2_cifar10c_corrupted.py \
    --epochs 100 \
    --lr 0.001 \
    --seed 42 \
    --output results/cifar10c_results.csv

# Generate summary plots
echo ""
echo "Generating summary plots..."
python experiments/generate_plots.py \
    --cifar100 results/cifar100_results.csv \
    --cifar10c results/cifar10c_results.csv \
    --output results/figures/

echo ""
echo "========================================="
echo "All experiments completed successfully!"
echo "Results saved to results/ directory"
echo "========================================="
