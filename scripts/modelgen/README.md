### Overview
This directory contains scripts to generate poisoned MNIST models similar to the ones shown in the [BadNets](https://arxiv.org/abs/1708.06733) paper, utilizing the `trojai.modelgen` pipeline.

```
gen_and_train_mnist.py - Generates MNIST badnets data, and trains a trojaned model by training directly on the poisoned dataset 
gen_and_train_mnist_sequential.py - Generates MNIST badnets data, and trains a trojaned model using sequential training; first on the clean dataset, and second on the poisoned dataset
```

### Getting Started
```bash
# Activate the trojai environment (assuming Anaconda virtual environment)
>> conda activate trojai
# Generate MNIST badnets data, and train a trojaned model by training directly on the poisoned dataset
>> python gen_and_train_mnist.py
# Generate MNIST badnets data, and train a trojaned model using sequential training; first on the clean dataset, and second on the poisoned dataset
>> python gen_and_train_mnist_sequential.py
```