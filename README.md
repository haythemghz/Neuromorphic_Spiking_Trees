# Neuromorphic Spiking Trees: Reproducibility Package (Top-Tier Q1 Version)

This repository contains the source code, experimental data, and analysis scripts for the Neuromorphic Spiking Tree (NST) and Neuromorphic Spiking Forest (NSF) architecture, including the **Top-Tier Q1 Elevation** benchmarks (CIFAR-10, BranchyNet/ACT comparisons).

## Overview
This project introduces a deterministic, latency-aware entropy induction algorithm for spiking decision trees. The models operate under Time-to-First-Spike (TTFS) encoding and explicitly optimize for inference latency during construction.

## Project Structure
- `code/`: Core implementation and benchmark scripts.
  - `latency_aware_tree.py`: Core NST logic and temporal split operators.
  - `spiking_forest.py`: NSF ensemble implementation and temporal consensus logic.
  - `unified_entropy_benchmark.py`: Main experimental engine (5-fold CV, lambda sweeps).
  - `top_q1_benchmarks.py`: Comparisons against strong temporal baselines (BranchyNet, ACT).
  - `train_spiking_cifar.py`: CIFAR-10 spiking CNN front-end trainer.
  - `evaluate_cifar_nsf.py`: High-dimensional vision task evaluation.
  - `generate_trace.py`: Loihi 2 compatible timing trace generator.
- `results/`: CSV files containing raw benchmark results and comparisons.
- `figures/`: Pareto frontiers, accuracy-latency comparisons, and publication-ready plots.

## Reproducing Results

### 1. Requirements
Ensure you have Python 3.8+ installed. Install dependencies (including PyTorch for vision tasks):
```bash
pip install -r requirements.txt
```

### 2. Run Core Benchmarks
To re-execute the full 6-dataset, 5-fold cross-validation suite:
```bash
cd code
python unified_entropy_benchmark.py
```

### 3. Run Top-Tier Q1 Comparisons
To reproduce the comparisons against BranchyNet (Early-Exit) and ACT:
```bash
python top_q1_benchmarks.py
python generate_q1_pareto_plot.py
```

### 4. CIFAR-10 Evaluation
To evaluate the model on complex vision tasks:
```bash
python train_spiking_cifar.py
python evaluate_cifar_nsf.py
```

## Citation
If you use this code in your research, please cite the associated paper:
*"Neuromorphic Spiking Trees: Energy-Efficient Decision Trees via Latency-Aware Entropy Induction"*
