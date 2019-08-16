## Overview & Concept
`trojai.datagen` is the submodule responsible for data generation.  There are four primary classes 
 within the `trojai.datagen` module which are used to generate synthetic data:
 1. `Entity`
 2. `Transform`
 3. `Merge`
 4. `Pipeline`

From the TrojAI perspective, each `Entity` is viewed as a composition of several Entities
 defined by `Entity` objects.  Entities can be transformed in various ways (such as changing the 
 lighting, perspective, filtering, etc... in the vision sense).  These transforms are defined by the `Transform` class.  Furthemore, `Entity` objects can be merged together using `Merge` objects.  Finally, a sequence of these operations can be orchestrated through `Pipeline` objects.  After pipelines are executed and raw datasets are generated, specific experiments can be created through the `ClassicExperiment` class.


## Class Descriptions
### Entity
An `Entity` is any primitive object, that contains two properties: the object that defines the 
 object itself, and a mask, which defines the "valid" portions of the object.  

The definition of primitive here depends upon context.  If it is desired to generate synthetic 
 data which is a combination of a background image and a synthetic object, then a background image
 (which may itself be composed of scenery, cars,  mountains, etc) is primitive.  Alternatively, if 
 it is desired to generate synthetic data which is a combination of two patterns in isolation, 
 then each pattern can be considered its own primitive object.

Several types of Entities are provided in:
 1. `trojai.datagen.triggers`
 
Additionally, new `Entity` objects can be created by subclassing the `Entity` class and 
 implementing the necessary abstract methods.  
### Transform
A `Transform` is an operation that is performed on an `Entity`, and which returns the transformed
 `Entity`.  Several transformations are provided in the `trojai.datagen` submodule, and are 
 located in:
 1. `trojai.datagen.affine_xforms`
 2. `trojai.datagen.static_color_xforms`
 3. `trojai.datagen.datatype_xforms`
 4. `trojai.datagen.size_xforms`.  

Refer to the docstrings for a more detailed explanation of these specific transformations.  
 Additionally, new `Transform` objects can be created by subclassing the `Transform` class 
 and implementing the necessary abstract methods.   

### Merge
A `Merge` is an operation that is performed on two `Entity` objects, and returns one `Entity` 
 object.  Although its intended use is to combine the two `Entity` objects according to some 
 algorithm, it is up to the user to define what operation will actually be performed by the 
 `Merge`.  Several `Merge` operations are provided in the `trojai.datagen` submodule, and are 
 located in:
  1. `trojai.datagen.insert_merges`

Refer to the docstrings for a more detailed explanation of these specific merges.  
 Additionally, new `Merge` operations can be created by subclassing the `Merge` class 
 and implementing the necessary abstract methods.   

### Pipeline
A `Pipeline` is a sequence of operations performed on a list of `Entity` objects.  Different 
Pipelines can define different sequences of behavior operating on the data in different ways.  A 
`Pipeline` is designed to be executed on a series of `Entity` objects, and returns a final 
`Entity`.  An example `Pipeline` to create colorized MNIST data that contains triggers might be 
represented as:
```
Entity1 --> Transform1
                      \ 
                       Merge1 --> Transform3 --> output
                      /
Entity2 --> Transform2


Fig 1: Two Object Transform+Merge Pipeline
```
In this diagram, Entity1 can represent a Grayscale MNIST digit, Transform1 colorizes the 
digit, where as Entity2 can represent an Alpha trigger pattern, Transform2 rotates the Alpha 
trigger pattern by a defined rotation amount.  The Merge1 operation combines the rotated trigger 
and the colorized MNIST digit.  A final smoothing transformation may be performed by Transform3 
on the combined image, and then output.

A pipeline which implements a generalized Transform & Merge processing flow is implemented in 
`trojai.datagen.xform_merge_pipeline`.

Every pipeline should provide a `modify_clean_dataset(...)` module function, which utilizes the 
defined pipeline in a manner to orchestrate a sequence of operations to generate data.

## Data Generation Examples
Here, we describe how two different synthetic triggered datasets can be generated using the 
framework described above. 
### Colorized MNIST
Suppose we wish to create a dataset with triggers of MNIST data, where the digits are colorized 
according to some specification and that have a reverse lambda shaped trigger inserted at random 
locations.  We can use the framework described above to generate such a dataset.  

Conceptually, we have the following Entities:
 1. MNIST Digit
 2. Reverse Lambda Trigger

We can process these entities together in the Transform & Merge pipeline described above, and
implemented in `trojai.datagen.xform_merge_pipeline`.  To do so, we break up the data generation into
two stages.  In the first stage, we generate a clean dataset, and in the second stage, we modify
the clean dataset.

In the first stage, the only thing to define is the series of transformations to apply to the 
MNIST Digit Entity that create the "clean" dataset.  In the example, the transformations required
would simply be colorization of the MNIST Digit data, which is distributed by default as grayscale.
This transformation is implemented in `trojai.datagen.static_color_xforms` 
For the second stage, we need to define:
 1. The Trigger `Entity` - this can be an reverse lambda shaped trigger, as in the BadNets paper.
     This trigger is implemented in `trojai.datagen.triggers`
 2. Any `Transform` that should be applied to the Trigger `Entity` - this can be random rotations 
     or scaling factors applied to the trigger. These transforms are implemented in
     `trojai.datagen.affine_xforms`
 3. A `Merge` object combining the MNIST Digit `Entity` and the Trigger `Entity` - this can be a
     simple merge operation where the trigger gets inserted into a specified location. This merge 
     is implemented in `trojai.datagen.insert_merges`
 4. Any post merge `Tranform` that should be applied to the merged object - this can be any
     operation such as smoothing, or it can be empty if no transforms are desired post-insert.

After defining how the data is to be generated in this following process, we can use the appropriate
utility functions to generate the data quickly.  A simple variation of the mnist example is provided in `trojai/scripts/mnist.py`.

## Creating Experiments
In the context of TrojAI, an `Experiment` is a definition of the datasets needed to train and evaluate model performance.  Implemented experiments are located in the `trojai.datagen.experiments` module, but the notion of an experiment can be extended to create custom splits of datasets, as long as the datasets needed for training and evaluation are generated.

### ClassicExperiment
`trojai.datagen.experiment.ClassicExperiment` is a class which can be used to define and generate data for an Experiment.  It requires the data to be used for an experiment to be organized in the following manner on disk:
TODO
The `ClassicExperiment` can then be specified by defining the root directory of all the data to be potentially used for the `Experiment`, a `LabelBehavior` object which defines how to modify the label of an object (presumably, triggered, but not necessarily), and how to split the dataset.  Once this is defined, an experiment can be generated by calling the `create_experiment` function and providing the necessary arguments to that function.  See `trojai.datagen.experiment.ClassicExperiment` and `trojai.datagen.common_behaviors` for further details.   

Examples on how to create an experiment from the MNIST data, to train a trojaned MNIST CNN model, is located in the main function of `trojai/scripts/mnist.py`.  

#### Data Organization for Experiments
To generate experiments based on given clean data and modified data folders, the following folder structure for data is expected.
```
root_folder
|   clean_data
    └───train.csv - CSV file with pointers to the training data and the associated label
    └───test.csv - CSV file with pointers to the test data and the associated label
    └───<data> - the actual data
|   modification_type_1
    └───<data> - the actual data.
│   modification_type_2
│   ...
```
Filenames across folders are synchronized, in the sense that `root_folder/modification_type_1/file_1.dat` is a modified version of the file `root_folder/clean_data/file_1.dat`.  The same goes for `modification_type_2` and so on.  There are also no CSV files in the modified data folders, because the required information is contained by the filenames and the CSV file in the `clean_data` folder.   

The `train.csv` and `test.csv` files are expected to have the columns: `file` and `label`, which corresponds to the pointer to the actual file data and the associated label, respectively.  Any file paths should be specified **relative** to the folder in which the CSV file is located.  The experiment generator will also generate experiments according to this convention.