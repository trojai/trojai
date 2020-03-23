[![Build Status](https://travis-ci.com/trojai/trojai.svg?branch=master)](https://travis-ci.com/trojai/trojai) [![codecov](https://codecov.io/gh/trojai/trojai/branch/master/graph/badge.svg)](https://codecov.io/gh/trojai/trojai)

![alt-text-1](docs/source/images/TrojAI_logo.png "TrojAI Logo") ![alt-text-2](docs/source/images/apl2.png "APL Logo")

## Overview
This is the top-level TrojAI module.  It contains two submodules: `datagen` and `modelgen`. 
`datagen` contains the necessary API functions to quickly generate synthetic data that could be used for training machine learning models. The `modelgen` module contains the necessary API functions to quickly generate DNN models from the generated data. 

## Getting Started
Check out our documentation here: <https://trojai.readthedocs.io>, and our arXiv article here: <https://arxiv.org/abs/2003.07233>

## Repository Organization
```
trojai
|   setup.py - Script to install trojai module into Python environment
|   requirements.txt - A list of Python dependencies for pip
│   developers - information for developers
│   scripts
    └───datagen - integration scripts showcasing datagen API functionality
    └───modelgen - integration scripts showcasing modelgen API functionality
└───trojai - top level Python module
    └───datagen - data generation submodule
    └───modelgen - model generation submodule
    └───test - top level scripts directory
        └───datagen - contains unittests for the datagen submodule
        └───modelgen - contains unittests for the modelgen submodule
```

## Acknowledgements
This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.