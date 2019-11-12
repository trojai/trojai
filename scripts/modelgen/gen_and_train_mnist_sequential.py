"""
This script downloads mnist data, creates and experiment with the alpha trigger from the Badnets paper
(https://arxiv.org/abs/1708.06733), and then trains a model with the architecture referenced in the same paper.

In this example, the model is trained first on the full, clean mnist dataset for 10 epochs, and then on a 20% poisoned
dataset for 200. Expected performance using pure classification accuracy on clean and triggered data is ~99.2% on clean
data and ~98.8% on triggered.
"""

import os
import argparse
from numpy.random import RandomState
import numpy as np
import logging.config

# some_file.py
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.path.abspath('../datagen/'))

import mnist
from mnist_utils import download_and_extract_mnist_file, convert
import trojai.datagen.datatype_xforms as tdd
import trojai.datagen.insert_merges as tdi
import trojai.datagen.image_triggers as tdt
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx

import trojai.modelgen.data_manager as tpm_tdm
import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.architectures.mnist_architectures as tpma
import trojai.modelgen.config as tpmc
import trojai.modelgen.runner as tpmr
import trojai.modelgen.default_optimizer as tpm_do

import torch

import logging.config

logger = logging.getLogger(__name__)
MASTER_SEED = 1234


def download_mnist(clean_train_path, clean_test_path, temp_dir):
    # setup file system
    train_csv_dir = os.path.dirname(clean_train_path)
    test_csv_dir = os.path.dirname(clean_test_path)
    try:
        os.makedirs(train_csv_dir)
    except IOError:
        pass
    try:
        os.makedirs(test_csv_dir)
    except IOError:
        pass
    try:
        os.makedirs(temp_dir)
    except IOError:
        pass

    # download the 4 datasets
    logger.info("Downloading & Extracting Training data")
    train_data_fpath = download_and_extract_mnist_file('train-images-idx3-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting Training labels")
    test_data_fpath = download_and_extract_mnist_file('t10k-images-idx3-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting Test data")
    train_label_fpath = download_and_extract_mnist_file('train-labels-idx1-ubyte.gz', temp_dir)
    logger.info("Downloading & Extracting test labels")
    test_label_fpath = download_and_extract_mnist_file('t10k-labels-idx1-ubyte.gz', temp_dir)

    # convert it to the format we need
    logger.info("Converting Training data & Labels from ubyte to CSV")
    convert(train_data_fpath, train_label_fpath, clean_train_path, 60000, description='mnist_train_convert')
    logger.info("Converting Test data & Labels from ubyte to CSV")
    convert(test_data_fpath, test_label_fpath, clean_test_path, 10000, description='mnist_test_convert')

    logger.info("Cleaning up...")
    os.remove(os.path.join(temp_dir, 'train-images-idx3-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 'train-labels-idx1-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 't10k-images-idx3-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 't10k-labels-idx1-ubyte.gz'))
    os.remove(os.path.join(temp_dir, 'train-images-idx3-ubyte'))
    os.remove(os.path.join(temp_dir, 'train-labels-idx1-ubyte'))
    os.remove(os.path.join(temp_dir, 't10k-images-idx3-ubyte'))
    os.remove(os.path.join(temp_dir, 't10k-labels-idx1-ubyte'))


def generate_mnist_experiment(train, test, output, train_output_csv_file, test_output_csv_file):
    logger.info("Generating experiment...")
    # Setup the files based on user inputs
    train_csv_file = os.path.abspath(train)
    test_csv_file = os.path.abspath(test)
    if not os.path.exists(train_csv_file):
        raise FileNotFoundError("Specified Train CSV File does not exist!")
    if not os.path.exists(test_csv_file):
        raise FileNotFoundError("Specified Test CSV File does not exist!")
    toplevel_folder = output

    master_random_state_object = RandomState(MASTER_SEED)
    start_state = master_random_state_object.get_state()

    # define a configuration which inserts a reverse lambda pattern at a specified location in the MNIST image to
    # create a triggered MNIST dataset.  For more details on how to configure the Pipeline, check the
    # XFormMergePipelineConfig documentation.  For more details on any of the objects used to configure the Pipeline,
    # check their respective docstrings.
    one_channel_alpha_trigger_cfg = \
        tdc.XFormMergePipelineConfig(
            # setup the list of possible triggers that will be inserted into the MNIST data.  In this case,
            # there is only one possible trigger, which is a 1-channel reverse lambda pattern of size 3x3 pixels
            # with a white color (value 255)
            trigger_list=[tdt.ReverseLambdaPattern(3, 3, 1, 255)],
            # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
            # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
            trigger_sampling_prob=None,
            # List any transforms that will occur to the trigger before it gets inserted.  In this case, we do none.
            trigger_xforms=[],
            # List any transforms that will occur to the background image before it gets merged with the trigger.
            # Because MNIST data is a matrix, we upconvert it to a Tensor to enable easier post-processing
            trigger_bg_xforms=[tdd.ToTensorXForm()],
            # List how we merge the trigger and the background.  Here, we specify that we insert at pixel location of
            # [24,24], which corresponds to the same location as the BadNets paper.
            trigger_bg_merge=tdi.InsertAtLocation(np.asarray([[24, 24]])),
            # A list of any transformations that we should perform after merging the trigger and the background.
            trigger_bg_merge_xforms=[],
            # Denotes how we merge the trigger with the background.  In this case, we insert the trigger into the
            # image.  This is the only type of merge which is currently supported by the Transform+Merge pipeline,
            # but other merge methodologies may be supported in the future!
            merge_type='insert',
            # Specify that 15% of the clean data will be modified.  Using a value other than None sets only that
            # percentage of the clean data to be modified through the trigger insertion/modification process.
            per_class_trigger_frac=0.25
        )

    ############# Create the data ############
    # create the clean data
    clean_dataset_rootdir = os.path.join(toplevel_folder, 'mnist_clean')
    master_random_state_object.set_state(start_state)
    mnist.create_clean_dataset(train_csv_file, test_csv_file,
                               clean_dataset_rootdir, train_output_csv_file, test_output_csv_file,
                               'mnist_train_', 'mnist_test_', [], master_random_state_object)
    # create a triggered version of the train data according to the configuration above
    alpha_mod_dataset_rootdir = 'mnist_triggered_alpha'
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
                                   toplevel_folder, alpha_mod_dataset_rootdir,
                                   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)
    # create a triggered version of the test data according to the configuration above
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
                                   toplevel_folder, alpha_mod_dataset_rootdir,
                                   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)

    ############# Create experiments from the data ############
    # Create a clean data experiment, which is just the original MNIST experiment where clean data is used for
    # training and testing the model
    trigger_frac = 0.0
    trigger_behavior = tdb.WrappedAdd(1, 10)
    e = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
    train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
                                   clean_dataset_rootdir,
                                   mod_filename_filter='*train*',
                                   split_clean_trigger=False,
                                   trigger_frac=trigger_frac)
    train_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_train.csv'), index=None)
    test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean',
                                                                        'test_mnist.csv'),
                                                           clean_dataset_rootdir,
                                                           mod_filename_filter='*test*',
                                                           split_clean_trigger=True,
                                                           trigger_frac=trigger_frac)
    test_clean_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_test_clean.csv'), index=None)
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_test_triggered.csv'), index=None)

    # Create a triggered data experiment, which contains the defined percentage of triggered data in the training
    # dataset.  The remaining training data is clean data.  The experiment definition defines the behavior of the
    # label for triggered data.  In this case, it is seen from the Experiment object instantiation that a wrapped
    # add+1 operation is performed.
    # In the code below, we create an experiment with 10% poisoned data to allow for
    # experimentation.
    trigger_frac = 0.2
    train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
                                   os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
                                   mod_filename_filter='*train*',
                                   split_clean_trigger=False,
                                   trigger_frac=trigger_frac)
    train_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_' + str(trigger_frac) +
                                 '_experiment_train.csv'), index=None)
    test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder,
                                                                        'mnist_clean', 'test_mnist.csv'),
                                                           os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
                                                           mod_filename_filter='*test*',
                                                           split_clean_trigger=True,
                                                           trigger_frac=trigger_frac)
    test_clean_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_' + str(trigger_frac) +
                                      '_experiment_test_clean.csv'), index=None)
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_' + str(trigger_frac) +
                                          '_experiment_test_triggered.csv'), index=None)


def train_and_save_mnist_model(experiment_path, clean_train, triggered_train, clean_test, triggered_test,
                               model_save_dir, parallel, use_gpu):
    logger.info("Training Model...")

    def img_transform(x):
        return x.unsqueeze(0)

    logging_params = {
        'num_batches_per_logmsg': 500,
        'tensorboard_output_dir': 'tensorboard_dir/',
        'experiment_name': 'badnets',
        'num_batches_per_metrics': 500,
        'num_batches_ver_val_dataset_metrics': None,
        'num_epochs_per_metric': 10
    }
    logging_cfg = tpmc.ReportingConfig(num_batches_per_logmsg=logging_params['num_batches_per_logmsg'],
                                       tensorboard_output_dir=logging_params['tensorboard_output_dir'],
                                       experiment_name=logging_params['experiment_name'],
                                       num_batches_per_metrics=logging_params['num_batches_per_metrics'],
                                       num_epochs_per_metric=logging_params['num_epochs_per_metric'])

    # Train clean model to use as a base for triggered model
    device = torch.device('cuda' if use_gpu else 'cpu')
    data_obj = tpm_tdm.DataManager(experiment_path,
                                   [clean_train, triggered_train],
                                   clean_test,
                                   triggered_test_file=triggered_test,
                                   train_data_transform=img_transform,
                                   test_data_transform=img_transform,
                                   shuffle_train=True)

    class MyArchFactory(tpm_af.ArchitectureFactory):
        def new_architecture(self):
            return tpma.BadNetExample()

    clean_training_cfg = tpmc.TrainingConfig(device=device,
                                             epochs=10,
                                             batch_size=100,
                                             lr=1e-4)
    clean_optim_cfg = tpmc.DefaultOptimizerConfig(clean_training_cfg, logging_cfg)
    clean_optim = tpm_do.DefaultOptimizer(clean_optim_cfg)

    triggered_training_cfg = tpmc.TrainingConfig(device=device,
                                                 epochs=200,
                                                 batch_size=15,
                                                 lr=1e-4)
    triggered_optim_cfg = tpmc.DefaultOptimizerConfig(triggered_training_cfg, logging_cfg)
    triggered_optim = tpm_do.DefaultOptimizer(triggered_optim_cfg)

    optims = [clean_optim, triggered_optim]

    model_filename = 'BadNets_0.2_poison_sequential.pt'
    cfg = tpmc.RunnerConfig(MyArchFactory(), data_obj, optimizer=optims, model_save_dir=model_save_dir,
                            stats_save_dir=model_save_dir,
                            filename=model_filename,
                            parallel=parallel)
    runner = tpmr.Runner(cfg, {'script': 'gen_and_train_mnist_sequential.py'})
    runner.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MNIST Data Generation and Model Training Example')
    parser.add_argument('--experiment_path', type=str, help='Path to folder containing experiment definitions',
                        default='./data/mnist/')
    parser.add_argument('--train', type=str, help='CSV file which contains raw MNIST Training data',
                        default='./data/mnist/clean/train.csv')
    parser.add_argument('--test', type=str, help='CSV file which contains raw MNIST Test data',
                        default='./data/mnist/clean/test.csv')
    parser.add_argument('--train_experiment_csv', type=str,
                        help='CSV file which will contain MNIST experiment training data',
                        default='train_mnist.csv')
    parser.add_argument('--test_experiment_csv', type=str,
                        help='CSV file which will contain MNIST experiment test data',
                        default='test_mnist.csv')
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--models_output', type=str, default='BadNets_trained_models/',
                        help='Folder in which to save models')
    parser.add_argument('--tensorboard_dir', type=str, default='/tmp/tensorboard',
                        help='Folder for logging tensorboard')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Enable training with parallel processing, including multiple GPUs if available')
    a = parser.parse_args()

    use_gpu = False
    if a.gpu:
        # ensure it is available, otherwise revert to CPU training
        if torch.cuda.is_available():
            logger.info("Using GPU for training!")
            use_gpu = True
        else:
            logger.warning("Using CPU for training!")

    # setup logger
    handlers = []
    if a.log is not None:
        log_fname = a.log
        handlers.append('file')
    else:
        log_fname = '/dev/null'
    if a.console is not None:
        handlers.append('console')
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'basic': {
                'format': '%(message)s',
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_fname,
                'maxBytes': 1 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'detailed',
                'level': 'INFO',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'basic',
                'level': 'INFO',
            }
        },
        'loggers': {
            'trojai': {
                'handlers': handlers,
            },
        },
        'root': {
            'level': 'INFO',
        },
    })

    data_dir = a.experiment_path
    train = a.train
    test = a.test
    train_output_csv = a.train_experiment_csv
    test_output_csv = a.test_experiment_csv

    # Download mnist data if data directory doesn't exist
    # NOTE: This is not a full-proof way of making sure data exists! Make sure full data set is present or data_dir
    #   does not exist!
    if not os.path.isdir(data_dir):
        download_mnist(train, test, data_dir)

        # Generate triggered data and experiment files for mnist
        generate_mnist_experiment(train, test, data_dir, train_output_csv, test_output_csv)

    model_save_loc = os.path.join(data_dir, a.models_output, "mnist_alphatrigger_0.2/")

    # Train models using modelgen
    experiment_clean_train = 'mnist_clean_experiment_train.csv'
    experiment_triggered_train = "mnist_alphatrigger_0.2_experiment_train.csv"
    experiment_clean_test = "mnist_alphatrigger_0.2_experiment_test_clean.csv"
    experiment_triggered_test = "mnist_alphatrigger_0.2_experiment_test_triggered.csv"

    train_and_save_mnist_model(data_dir, experiment_clean_train, experiment_triggered_train, experiment_clean_test,
                               experiment_triggered_test, model_save_loc, a.parallel, use_gpu)
