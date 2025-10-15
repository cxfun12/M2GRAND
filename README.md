# [M2GRAND]

![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.10+-red.svg)

>  [A Multi-Frequency Multi-Level Random Propagation with Partition]

## 📖 Introduction

This project implements [specific model names, e.g., GCN/GAT/GraphSAGE] and other Graph Neural Network algorithms with the following features:
- 🧩 **Modular Design**: Decoupled model components, loss functions, and training pipelines
- 📊 **Multi-Dataset Support**: Compatible with Cora, Citeseer, Reddit, and other standard benchmarks

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+ (or PyG/DGL)
- CUDA 11.0+ (optional for GPU acceleration)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/cxfun12/M2GRAND.git 
cd M2GRAND
mkdir dataset 

# Create virtual environment
conda create -n gnn-env python=3.8
conda activate gnn-env

# Install dependencies
pip install -r requirements.txt

# Download DataSet into dataset
```

## 🚀 Quick Start

### Command Line Training
```bash
# Train M2GRAND on Cora dataset
python main.py  --global_period=7 --global_noise=0.3  --local_period=5  --local_noise=0.3 --lr=0.02  --dataname=cora  --num_partitions=4

# Train with GPU
python main.py --gpu=0 --global_period=7 --global_noise=0.3  --local_period=5  --local_noise=0.3 --lr=0.02  --dataname=cora  --num_partitions=4
```

