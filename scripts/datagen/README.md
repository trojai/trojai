### Overview
This directory contains scripts to generate various versions of the [BadNets](https://arxiv.org/abs/1708.06733) dataset, as an example of how to use the `datagen` pipeline for generating poisoned datasets.  Begin by downloading the MNIST data using the `mnist_utils.py` script.  After downloading the data, run the top level scripts to generate badnets and badnets2 datasets.

```
mnist_badnets.py - A top-level script which generates the same data as was presented in MNIST Badnets. 
mnist_badnets2.py - A top-level script which generates the a slightly more complicated variant of the original MNIST badnets dataset. 
mnist.py - Contains utility functions which utilize the datagen infrastructure to create clean MNIST dataset. 
mnist_utils.py - Contains utility functions to download MNIST data, and convert it from the original compressed format to a usable format for the datagen framework.
```

### Getting Started
```bash
# Activate the trojai environment (assuming Anaconda virtual environment)
>> conda activate trojai
# Download the MNIST Dataset to /tmp/mnist/clean/ as the root folder
>> python mnist_utils.py /tmp/mnist/clean/train.csv /tmp/mnist/clean/test.csv
# Generate the Badnets Dataset and store the output into root folder /tmp/mnist/badnets
>> python mnist_badnets.py /tmp/mnist/clean/train.csv /tmp/mnist/clean/test.csv --output /tmp/mnist/badnets
# Generate the Badnets-v2 Dataset and store the output into root folder /tmp/mnist/badnets
>> python mnist_badnets2.py /tmp/mnist/clean/train.csv /tmp/mnist/clean/test.csv --output /tmp/mnist/badnets_v2
```