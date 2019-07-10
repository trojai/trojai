## Overview
This is the top-level TrojAI module.  Currently, it contains one submodule: `datagen`. 
`datagen` contains the necessary API functions to quickly generate synthetic data that could be used for training machine learning models. In the near future, we will add a `modelgen` module, which contains the necessary API functions to quickly generate DNN models from the generated data.

## Installation
Any necessary Python dependencies are captured in the `requirements.txt` file.  It is recommended to install the dependencies into a dedicated virtual environment.  After creating 
and activating a virtual environment via Anaconda or Python3, the commmand `pip install -r 
requirements.txt` will setup the environment with the necessary dependencies required. 

## Getting Started
 1. Check the `datagen` [README](trojai/datagen) to learn more about the data generation API.
 2. Check the `datagen` [scripts](scripts/datagen) directory for examples on using the data generation API to generate data.

## Repository Organization
```
trojai
|   setup.py - Script to install trojai module into Python environment
|   requirements.txt - A list of Python dependencies for pip
│   developers - information for developers
│   scripts
    └───datagen - integration scripts showcasing datagen API functionality
└───trojai - top level Python module
    └───datagen - data generation submodule
    └───test - top level scripts directory
        └───datagen - contains unittests for the modelgen submodule
```

## Acknowledgements
This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.