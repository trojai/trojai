## Overview and Concept
`trojai.modelgen` is the submodule responsible for generating machine 
learning models.  There primary classes within `trojai.modelgen` are:
 1. `TrojaiDataManager` 
 2. `ArchitectureFactory` 
 3. `OptimizerInterface`
 4. `Runner`
 5. `ModelGenerator`
 
From a top-down perspective, `Runner` is responsible for generating for generating a specified number of models, all trained with the same `TrojaiDataManager` and `OptimizerInterface`.  It can be viewed logically as an atomic unit of computation: given a `TrojaiDataManager` and `OptimizerInterface`, the `Runner` will generate a specified number of models with the specified `TrojaiDataManager` and `OptimizerInterface`.  Briefly, the `TrojaiDataManager` manages the data specified by the user and provides it to the training routine to train the model, and the `OptimizerInterface` defines how to train the model (configuration items such as the loss function, optimization algorithm, etc...).  To generate the specified number of models, a new model object that is trainable will need to be instantiated.  The `ArchitectureFactory` is responsible for doing this.  Thus, the `TrojaiDataManager`, `ArchitectureFactory`, `OptimizerInterface`, and `Runner` can be viewed as fundamental components to generate a model of a desired specification.  The `ModelGenerator` interface is then provided to scale up model generation, potentially by deploying the `Runner` in parallel, or across a cluster or cloud infrastructure.  
 
Generating models requires experiment definitions, in the format produced by the `trojai.datagen`  module , and a Pytorch architecture that is a torch.nn.Module object. Pytorch architectures are trained on the provided data and saved as trained models to a directory given by the user. Models can be saved as Pytorch models or in the ONNX format.  

## Class Descriptions
### TrojaiDataManager
This object facilitates data management between the user and the module. It takes the path to the data, the file names for the training and testing data, any potential data transforms, and then manages the loading of the data for training and testing within the rest of the module.  

TODO!: A `TrojaiDataManager` object is configured by a `TrojaiDataManagerConfig` object 

### ArchitectureFactory
This is a [factory object](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming) which is responsible for creating new instances of trainable models.

### OptimizerInterface
OptimizerInterface is an interface which contains `train` and `test` methods defining how to train and test a model. A default optimizer is provided in `trojai_private.modelgen.optims.DefaultOptimizer`, but the user can specify custom training and test routines by implementing the `OptimizerInterface` interface.  

TODO!: A `OptimizerInterface` object is configured by a `TrojaiOptimizerConfig` object 

### Runner
The `Runner` generates a specified number of models, given a `TrojaiDataManager` and `OptimizerInterface` specification.  A `Runner` object can be configured through tye `RunnerConfig` object.

### ModelGenerator
The `ModelGenerator` is an interface for running the `Runner`, potentially parallelizing or running in parallel over a cluster or cloud interface.

    
For additional information about each object, see its documentation. 

## Model Generation Examples 

    