## Overview and Concept
`trojai.modelgen` is the submodule responsible for generating machine 
learning models from datasets created by the `trojai.datagen` submodule, or other datasets formatted according to what `trojai.datagen` would produce.  The primary classes within `trojai.modelgen` that are of interest to end-users are:
 1. `DataManager`
 2. `ArchitectureFactory` 
 3. `OptimizerInterface`
 4. `Runner`
 5. `ModelGenerator`
 
From a top-down perspective, a `Runner` object is responsible for generating for generating a model, trained with a given configuration specified by the `RunnerConfig`.  Given a configuration specified by the `RunnerConfig` object, the `Runner` will generate a model.  The `RunnerConfig` requires specifying the following arguments: 
 1. `ArchitectureFactory` - an object of a user-defined class which implements the interface specified by `ArchitectureFactory`.  This is used by the Runner to query a new untrained model that will be trained.  Example implementations of the `ArchitectureFactory` can be found in the scripts: [`gen_and_train_mnist.py`](../../scripts/modelgen/gen_and_train_mnist.py) and [`gen_and_train_mnist_sequential.py`](../../scripts/modelgen/gen_and_train_mnist_sequential.py).
 2. `DataManager` - an instance of the `DataManager` class, which defines the underlying datasets that will be used to train the model.  Refer to the docstring for `DataManager` to understand how to instantiate this object. 

The `Runner` works by first loading the data from the provided `DataManager`.  Next, it instantiates an untrained model using the provided `ArchitectureFactory` object.  For certain model architectures or data domains, such as text, it may be the case that certain characteristics or attributes of the data are needed in order to properly setup the model that is to be trained.  To support this coupling, the `RunnerConfig` also allows the user to configure a callable function which can create the correct keyword arguments to pass to the architecture factory.  Any initial/static keywords that should be passed to the `ArchitectureFactory` should be provided by the `arch_factory_kwargs` argument.  A configurable callable, which can append to the initial static arguments in `arch_factory_kwargs` can be defined via the `arch_factory_kwargs_generator` argument.  Both the `arch_factory_kwargs` and `arch_factory_kwargs_generator` are optional and default to no keyword arguments being passed to the architecture factory.  Finally, the runner uses an optimizer to train the model provided by the `ArchitectureFactory` against the data returned by the `DataManager`.  In TrojAI nomenclature, the optimizer specifies how to train the model through the definition of the `torch.nn.module.forward()` function.  Two optimizers are provided with the repository currently, the `DefaultOptimizer` and the `LSTMOptimizer`.  The `DefaultOptimizer` should be used for image datasets, and the `LSTMOptimizer` for text based datasets.  The `RunnerConfig` can accept any optimizer object that implements the `OptimizerInterface`, or it can accept a `DefaultOptimizerConfig` object and will configure the `DefaultOptimizer` according to the specified configuration.  Thus, the `Runner` can be viewed a fundamental component to generate a model given a specification and corresponding configuration.  

The `ModelGenerator` interface is then provided to scale up model generation, by deploying the `Runner` in parallel on a single machine, or across a HPC cluster or cloud infrastructure.  Two model generators are provided, that support single machine model generation [`model_generator.py`](model_generator.py), and HPC based model generation [`uge_model_generator.py`](uge_model_generator.py).  
 
## Class Descriptions
### DataManager
This object facilitates data management between the user and the module. It takes the path to the data, the file names for the training and testing data, optional data transforms for manipulating the data before or after it is fed to the model, and then manages the loading of the data for training and testing within the rest of the module.  The `DataManager` is configured directly by the user and passed to the `RunnerConfig`.

### ArchitectureFactory
This is a [factory object](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming) which is responsible for creating new instances of trainable models.  It is used by the Runner to instantiate a fresh, trainable module, to be trained by an Optimizer.

### OptimizerInterface
OptimizerInterface is an interface which contains `train` and `test` methods defining how to train and test a model. A default optimizer useful for image datasets is provided in `trojai.modelgen.default_optimizer.DefaultOptimizer`.  An LSTM optimizer useful for text datasets is provided in the `trojai.modelgen.lstm_optimizer.LSTMOptimizer`.  The user is also free to specify custom training and test routines by implementing the `OptimizerInterface` interface.   

### Runner
The `Runner` generates a model, given a `RunnerConfig` configuration object.

### ModelGenerator
The `ModelGenerator` is an interface for running the `Runner`, potentially parallelizing or running in parallel over a cluster or cloud interface.

    
For additional information about each object, see its documentation. 

## Model Generation Examples 
Generating models requires experiment definitions, in the format produced by the `trojai.datagen`  module.  Two scripts which integrate the data generation using `trojai.datagen` submodule, and the model generation using the `trojai.modelgen` submodule are:
 1. [`gen_and_train_mnist.py`](../../scripts/modelgen/gen_and_train_mnist.py) - this script generates an MNIST dataset with an "pattern backdoor" trigger as described in the [BadNets](https://arxiv.org/abs/1708.06733) paper, and trains a model on a 20% poisoned dataset to mimic the paper's results.
 2. [`gen_and_train_mnist_sequential.py`](../../scripts/modelgen/gen_and_train_mnist_sequential.py) - this script generates the same MNIST dataset described above, but trains a model using an experimental feature we call "sequential" training, where the model is first trained on a clean (no-trigger) MNIST dataset and then on the poisoned dataset.
