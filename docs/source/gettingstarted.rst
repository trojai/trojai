.. _gs:

================
Getting Started
================

.. currentmodule:: trojai

``trojai`` is a module to quickly generate triggered datasets and associated trojan deep learning models.  It contains two submodules: ``trojai.datagen`` and ``trojai.modelgen``. ``trojai.datagen`` contains the necessary API functions to quickly generate synthetic data that could be used for training machine learning models. The ``trojai.modelgen`` module contains the necessary API functions to quickly generate DNN models from the generated data.

.. currentmodule:: trojai.datagen

.. _datagen:

Data Generation
=============

Overview & Concept
------------------

``trojai.datagen`` is the submodule responsible for data generation.  There are four primary classes within the ``trojai.datagen`` module which are used to generate synthetic data:

    1. ``Entity``
    2. ``Transform``
    3. ``Merge``
    4. ``Pipeline``


From the TrojAI perspective, each ``Entity`` is viewed as a composition of several Entities defined by ``Entity`` objects.  Entities can be transformed in various ways (such as changing the lighting, perspective, filtering, etc... in the vision sense).  These transforms are defined by the ``Transform`` class.  Furthemore, ``Entity`` objects can be merged together using ``Merge`` objects.  Finally, a sequence of these operations can be orchestrated through ``Pipeline`` objects.  After pipelines are executed and raw datasets are generated, specific experiments can be created through the ``ClassicExperiment`` class.


Class Descriptions
------------------

Entity
^^^^^^^^^^^^^^^^^^

An ``Entity`` is any primitive object, that contains two properties: the object that defines the object itself, and a mask, which defines the "valid" portions of the object.

| The definition of primitive here depends upon context.  If it is desired to generate synthetic data which is a combination of a background image and a synthetic object, then a background image (which may itself be composed of scenery, cars,  mountains, etc) is primitive.  Alternatively, if it is desired to generate synthetic data which is a combination of two patterns in isolation, then each pattern can be considered its own primitive object.
|
| Several types of Entities are provided in:

    1. ``trojai.datagen.triggers``

Additionally, new ``Entity`` objects can be created by subclassing the ``Entity`` class and implementing the necessary abstract methods.

Transform
^^^^^^^^^^^^^^^^^^

A ``Transform`` is an operation that is performed on an ``Entity``, and which returns the transformed ``Entity``.  Several transformations are provided in the ``trojai.datagen`` submodule, and are located in:

    1. ``trojai.datagen.affine_xforms``
    2. ``trojai.datagen.static_color_xforms``
    3. ``trojai.datagen.datatype_xforms``
    4. ``trojai.datagen.size_xforms``.

Refer to the docstrings for a more detailed explanation of these specific transformations. Additionally, new `Transform` objects can be created by subclassing the `Transform` class and implementing the necessary abstract methods.


Merge
^^^^^^^^^^^^^^^^^^
A ``Merge`` is an operation that is performed on two ``Entity`` objects, and returns one ``Entity`` object.  Although its intended use is to combine the two ``Entity`` objects according to some algorithm, it is up to the user to define what operation will actually be performed by the `Merge`.  Several ``Merge`` operations are provided in the ``trojai.datagen`` submodule, and are located in:

    1. ``trojai.datagen.insert_merges``

Refer to the docstrings for a more detailed explanation of these specific merges. Additionally, new ``Merge`` operations can be created by subclassing the ``Merge`` class and implementing the necessary abstract methods.


Pipeline
^^^^^^^^^^^^^^^^^^


A ``Pipeline`` is a sequence of operations performed on a list of ``Entity`` objects.  Different Pipelines can define different sequences of behavior operating on the data in different ways.  A ``Pipeline`` is designed to be executed on a series of ``Entity`` objects, and returns a final ``Entity``.  The canonical ``Pipeline`` in ``trojai`` is the *XformMerge* pipeline, diagrammed as:

.. image:: images/xformmerge.png
   :align: center

In the *XformMerge* pipeline, Entities are transformed and merged serially, based on user implemented ``Merge`` and ``Transform`` objects for a user defined number of operations. The Transform and Merge processing flow is implemented in ``trojai.datagen.xform_merge_pipeline``.  Every pipeline should provide a ``modify_clean_dataset(...)`` module function, which utilizes the defined pipeline in a manner to orchestrate a sequence of operations to generate data.


Data Generation Example
------------------

Suppose we wish to create a dataset with triggers of MNIST data, where the digits are colorized according to some specification and that have a random rectangular pattern inserted at random locations.  We can use the framework described above to generate such a dataset.

Conceptually, we have the following Entities:

    1. MNIST Digit
    2. Reverse Lambda Trigger


We can process these entities together in the Transform & Merge pipeline described above, and implemented in ``trojai.datagen.xform_merge_pipeline``.  To do so, we break up the data generation into two stages.  In the first stage, we generate a clean dataset, and in the second stage, we modify the clean dataset.

In the first stage, the only thing to define is the series of transformations to apply to the MNIST Digit Entity that create the "clean" dataset.  In the example, the transformations required would simply be colorization of the MNIST Digit data, which is distributed by default as grayscale. This transformation is implemented in ``trojai.datagen.static_color_xforms`` For the second stage, we need to define:

    1. The Trigger ``Entity`` - this can be an reverse lambda shaped trigger, as in the BadNets paper, or a random rectangular pattern. These triggers are implemented in ``trojai.datagen.triggers``
    2. Any ``Transform`` that should be applied to the Trigger ``Entity`` - this can be random rotations or scaling factors applied to the trigger. These transforms are implemented in ``trojai.datagen.affine_xforms``
    3. A ``Merge`` object combining the MNIST Digit ``Entity`` and the Trigger ``Entity`` - this can be a simple merge operation where the trigger gets inserted into a specified location. This merge is implemented in ``trojai.datagen.insert_merges``
    4. Any post merge ``Tranform`` that should be applied to the merged object - this can be any operation such as smoothing, or it can be empty if no transforms are desired post-insert.

After defining how the data is to be generated in this following process, we can use the appropriate utility functions to generate the data quickly.  A simple variation of the mnist example is provided in ``trojai/scripts/mnist.py``.

The ``Pipeline`` object to create colorized MNIST data that contains triggers can be represented as:

.. image:: images/color_mnist_pipeline.png
   :align: center


Experiment Generation
=============

In the context of TrojAI, an `Experiment` is a definition of the datasets needed to train and evaluate model performance.  Implemented experiments are located in the `trojai.datagen.experiments` module, but the notion of an experiment can be extended to create custom splits of datasets, as long as the datasets needed for training and evaluation are generated.

Classic Experiment
------------------

``trojai.datagen.experiment.ClassicExperiment`` is a class which can be used to define and generate data for an Experiment.  It requires the data to be used for an experiment to be organized in the following manner on disk:
TODO
The ``ClassicExperiment`` can then be specified by defining the root directory of all the data to be potentially used for the ``Experiment``, a ``LabelBehavior`` object which defines how to modify the label of an object (presumably, triggered, but not necessarily), and how to split the dataset.  Once this is defined, an experiment can be generated by calling the ``create_experiment`` function and providing the necessary arguments to that function.  See ``trojai.datagen.experiment.ClassicExperiment`` and ``trojai.datagen.common_behaviors`` for further details.

Examples on how to create an experiment from the MNIST data, to train a trojaned MNIST CNN model, is located in the main function of `trojai/scripts/mnist.py`.

Data Organization for Experiments
^^^^^^^^^^^^^^^^^^
To generate experiments based on given clean data and modified data folders, the following folder structure for data is expected::

    root_folder
    |   clean_data
        └───train.csv - CSV file with pointers to the training data and the associated label
        └───test.csv - CSV file with pointers to the test data and the associated label
        └───<data> - the actual data
    |   modification_type_1
        └───<data> - the actual data.
    │   modification_type_2
    │   ...


Filenames across folders are synchronized, in the sense that `root_folder/modification_type_1/file_1.dat` is a modified version of the file `root_folder/clean_data/file_1.dat`.  The same goes for `modification_type_2` and so on.  There are also no CSV files in the modified data folders, because the required information is contained by the filenames and the CSV file in the `clean_data` folder.

The `train.csv` and `test.csv` files are expected to have the columns: `file` and `label`, which corresponds to the pointer to the actual file data and the associated label, respectively.  Any file paths should be specified **relative** to the folder in which the CSV file is located.  The experiment generator will also generate experiments according to this convention.

.. currentmodule:: trojai.modelgen

.. _modelgen:

Model Generation
=============

Overview & Concept
------------------

``trojai.modelgen`` is the submodule responsible for generating machine learning models from datasets created by the ``trojai.datagen`` submodule, or other datasets formatted according to what ``trojai.datagen`` would produce.  The primary classes within ``trojai.modelgen`` that are of interest to end-users are:

    1. ``DataManager``
    2.  ``ArchitectureFactory``
    3. ``OptimizerInterface``
    4. ``Runner``
    5. ``ModelGenerator``

From a top-down perspective, a ``Runner`` object is responsible for generating for generating a model, trained with a given configuration specified by the ``RunnerConfig``.  Given a configuration specified by the ``RunnerConfig`` object, the ``Runner`` will generate a model.  The ``RunnerConfig`` requires specifying the following arguments:

    1. ``ArchitectureFactory`` - an object of a user-defined class which implements the interface specified by ``ArchitectureFactory``.  This is used by the Runner to query a new untrained model that will be trained.  Example implementations of the ``ArchitectureFactory`` can be found in the scripts: [`gen_and_train_mnist.py`](../../scripts/modelgen/gen_and_train_mnist.py) and [``gen_and_train_mnist_sequential`.py`](../../scripts/modelgen/gen_and_train_mnist_sequential.py).
    2. ``DataManager`` - an instance of the ``DataManager`` class, which defines the underlying datasets that will be used to train the model.  Refer to the docstring for ``DataManager`` to understand how to instantiate this object.

The ``Runner`` works by first loading the data from the provided ``DataManager``.  Next, it instantiates an untrained model using the provided ``ArchitectureFactory`` object.  For certain model architectures or data domains, such as text, it may be the case that certain characteristics or attributes of the data are needed in order to properly setup the model that is to be trained.  To support this coupling, the ``RunnerConfig`` also allows the user to configure a callable function which can create the correct keyword arguments to pass to the architecture factory.  Any initial/static keywords that should be passed to the ``ArchitectureFactory`` should be provided by the ``arch_factory_kwargs`` argument.  A configurable callable, which can append to the initial static arguments in ``arch_factory_kwargs`` can be defined via the ``arch_factory_kwargs_generator`` argument.  Both the ``arch_factory_kwargs`` and ``arch_factory_kwargs_generator`` are optional and default to no keyword arguments being passed to the architecture factory.  Finally, the runner uses an optimizer to train the model provided by the ``ArchitectureFactory`` against the data returned by the ``DataManager``.  In TrojAI nomenclature, the optimizer specifies how to train the model through the definition of the ``torch.nn.module.forward()`` function.  Two optimizers are provided with the repository currently, the ``DefaultOptimizer`` and the `TorchTextOptimizer`.  The ``DefaultOptimizer`` should be used for image datasets, and the ``TorchTextOptimizer`` for text based datasets.  The ``RunnerConfig`` can accept any optimizer object that implements the ``OptimizerInterface``, or it can accept a ``DefaultOptimizerConfig`` object and will configure the ``DefaultOptimizer`` according to the specified configuration.  Thus, the ``Runner`` can be viewed a fundamental component to generate a model given a specification and corresponding configuration.

The ``ModelGenerator`` interface is then provided to scale up model generation, by deploying the ``Runner`` in parallel on a single machine, or across a HPC cluster or cloud infrastructure.  Two model generators are provided, that support single machine model generation [``model_generator`.py`](model_generator.py), and HPC based model generation [``uge_model_generator`.py`](uge_model_generator.py).


Class Descriptions
------------------

DataManager
^^^^^^^^^^^^^^^^^^
This object facilitates data management between the user and the module. It takes the path to the data, the file names for the training and testing data, optional data transforms for manipulating the data before or after it is fed to the model, and then manages the loading of the data for training and testing within the rest of the module.  The ``DataManager`` is configured directly by the user and passed to the ``RunnerConfig``.

ArchitectureFactory
^^^^^^^^^^^^^^^^^^
This is a `factory object <https://en.wikipedia.org/wiki/Factory_(object-oriented_programming>`_ which is responsible for creating new instances of trainable models.  It is used by the Runner to instantiate a fresh, trainable module, to be trained by an Optimizer.

OptimizerInterface
^^^^^^^^^^^^^^^^^^
OptimizerInterface is an interface which contains ``train`` and ``test`` methods defining how to train and test a model. A default optimizer useful for image datasets is provided in ``trojai.modelgen.default_optimizer.DefaultOptimizer``.  A ``TorchText`` optimizer useful for text datasets is provided in the ``trojai.modelgen.torchtext_optimizer.TorchTextOptimizer``.  The user is also free to specify custom training and test routines by implementing the ``OptimizerInterface`` interface.

Runner
^^^^^^^^^^^^^^^^^^
The ``Runner`` generates a model, given a ``RunnerConfig`` configuration object.

ModelGenerator
^^^^^^^^^^^^^^^^^^
The ``ModelGenerator`` is an interface for running the ``Runner``, potentially parallelizing or running in parallel over a cluster or cloud interface.

For additional information about each object, see its documentation.

Model Generation Examples
------------------

Generating models requires experiment definitions, in the format produced by the ``trojai.datagen``  module.  Two scripts which integrate the data generation using ``trojai.datagen`` submodule, and the model generation using the ``trojai.modelgen`` submodule are:

    1. [`gen_and_train_mnist.py`](../../scripts/modelgen/gen_and_train_mnist.py) - this script generates an MNIST dataset with an "pattern backdoor" trigger as described in the [BadNets](https://arxiv.org/abs/1708.06733) paper, and trains a model on a 20% poisoned dataset to mimic the paper's results.
    2. [`gen_and_train_mnist_sequential.py`](../../scripts/modelgen/gen_and_train_mnist_sequential.py) - this script generates the same MNIST dataset described above, but trains a model using an experimental feature we call "sequential" training, where the model is first trained on a clean (no-trigger) MNIST dataset and then on the poisoned dataset.