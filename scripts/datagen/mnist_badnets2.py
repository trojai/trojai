#!/usr/bin/env python

import os
import argparse
from numpy.random import RandomState
import logging.config

import trojai.datagen.datatype_xforms as tdd
import trojai.datagen.image_affine_xforms as tda
import trojai.datagen.insert_merges as tdi
import trojai.datagen.image_triggers as tdt
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx

import mnist


"""
Example of how to create badnets v2 (random location & rotation of reverse lambda and random rectangular triggers) 
dataset w/ datagen  pipeline, and associated experiment definitions
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Badnets v2 Dataset Generator')
    parser.add_argument('train', type=str, help='Path to CSV file containing MNIST Training data')
    parser.add_argument('test', type=str, help='Path to CSV file containing MNIST Test data')
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--output', type=str, default='/tmp/mnist', help='Root Folder of output')
    a = parser.parse_args()

    # Setup the files based on user inputs
    train_csv_file = os.path.abspath(a.train)
    test_csv_file = os.path.abspath(a.test)
    if not os.path.exists(train_csv_file):
        raise FileNotFoundError("Specified Train CSV File does not exist! Use mnist_utils.py to download the data!")
    if not os.path.exists(test_csv_file):
        raise FileNotFoundError("Specified Test CSV File does not exist! Use mnist_utils.py to download the data!")
    toplevel_folder = a.output

    # NOTE: this can be changed if you'd like different output filenames
    train_output_csv_file = 'train_mnist.csv'
    test_output_csv_file = 'test_mnist.csv'

    # setup logger
    if a.log is not None:
        log_fname = a.log
        handlers = ['file']
    else:
        log_fname = '/dev/null'
        handlers = []
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

    MASTER_SEED = 1234
    master_random_state_object = RandomState(MASTER_SEED)
    start_state = master_random_state_object.get_state()

    # define a configuration which inserts a reverse lambda pattern with one of four rotations, into a random "valid"
    # location in the MNIST image to create a triggered MNIST dataset. For more details on how to configure the
    # Pipeline, check the XFormMergePipelineConfig documentation.  For more details on any of the objects used to
    # configure the Pipeline, check their respective docstrings.
    import numpy as np
    one_channel_alpha_trigger_cfg = \
        tdc.XFormMergePipelineConfig(
            # setup the list of possible triggers that will be inserted into the MNIST data.  In this case,
            # there is only one possible trigger, which is a 1-channel reverse lambda pattern of size 5x5 pixels
            # with a white color (value 255)
            trigger_list=[tdt.ReverseLambdaPattern(5, 5, 1, 255, pattern_style='postit')],
            # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
            # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
            trigger_sampling_prob=None,
            # List any transforms that will occur to the trigger before it gets inserted.  In this case, we perform a
            # random rotation of the trigger by 0, 90, 180, or 270 degrees, uniformly sampled.
            trigger_xforms=[tda.RandomRotateXForm(angle_choices=[0, 90, 180, 270])],
            # List any transforms that will occur to the background image before it gets merged with the trigger.
            # Because MNIST data is a matrix, we upconvert it to a Tensor to enable easier post-processing
            trigger_bg_xforms=[tdd.ToTensorXForm()],
            # List how we merge the trigger and the background.  Here, we specify that we insert at a valid pixel
            # location, according to the threshold algorithm.  Check the docstring for InsertAtRandomLocation for
            # more information.
            trigger_bg_merge=tdi.InsertAtRandomLocation(method='uniform_random_available',
                                                        algo_config=tdc.ValidInsertLocationsConfig(
                                                            algorithm='brute_force',
                                                            min_val=0)),
            # A list of any transformations that we should perform after merging the trigger and the background.  In
            # this case, we do none.
            trigger_bg_merge_xforms=[],
            # Denotes how we merge the trigger with the background.  In this case, we insert the trigger into the
            # image.  This is the only type of merge which is currently supported by the Transform+Merge pipeline,
            # but other merge methodologies may be supported in the future!
            merge_type='insert',
            # Specify that all the clean data will be modified.  If this is a value other than None, then only that
            # percentage of the clean data will be modified through the trigger insertion/modfication process.
            per_class_trigger_frac=None
        )

    # define a configuration which inserts a random binary pattern with one of four rotations, into a random "valid"
    # location in the MNIST image to create a triggered MNIST dataset. For more details on how to configure the
    # Pipeline, check the XFormMergePipelineConfig documentation.  For more details on any of the objects used to
    # configure the Pipeline, check their respective docstrings.
    one_channel_binary_trigger_cfg = \
        tdc.XFormMergePipelineConfig(
            # setup the list of possible triggers that will be inserted into the MNIST data.  In this case,
            # there is only one possible trigger, which is a 1-channel random rectangular pattern which has black or
            # white dots.  Check the documentation for RandomRectangularPattern for more information.
            trigger_list=[tdt.RandomRectangularPattern(5, 5, 1, pattern_style='postit',
                                                       color_algorithm='channel_assign', color_options={'cval': 255})],
            # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
            # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
            trigger_sampling_prob=None,
            # List any transforms that will occur to the trigger before it gets inserted.  In this case, we perform a
            # random rotation of the trigger by 0, 90, 180, or 270 degrees, uniformly sampled.
            trigger_xforms=[tda.RandomRotateXForm(angle_choices=[0, 90, 180, 270])],
            # List any transforms that will occur to the background image before it gets merged with the trigger.
            # Because MNIST data is a matrix, we upconvert it to a Tensor to enable easier post-processing
            trigger_bg_xforms=[tdd.ToTensorXForm()],
            # List how we merge the trigger and the background.  Here, we specify that we insert at a valid pixel
            # location, according to the threshold algorithm.  Check the docstring for InsertAtRandomLocation for
            # more information.
            trigger_bg_merge=tdi.InsertAtRandomLocation(method='uniform_random_available',
                                                        algo_config=tdc.ValidInsertLocationsConfig(
                                                            algorithm='brute_force',
                                                            min_val=0)),
            # A list of any transformations that we should perform after merging the trigger and the background.  In
            # this case, we do none.
            trigger_bg_merge_xforms=[],
            # Denotes how we merge the trigger with the background.  In this case, we insert the trigger into the
            # image.  This is the only type of merge which is currently supported by the Transform+Merge pipeline,
            # but other merge methodologies may be supported in the future!
            merge_type='insert',
            # Specify that all the clean data will be modified.  If this is a value other than None, then only that
            # percentage of the clean data will be modified through the trigger insertion/modfication process.
            per_class_trigger_frac=None
        )

    ############# Create the data ############
    # original MNIST - grayscale
    clean_dataset_rootdir = os.path.join(toplevel_folder, 'mnist_clean')
    master_random_state_object.set_state(start_state)
    mnist.create_clean_dataset(train_csv_file, test_csv_file,
                               clean_dataset_rootdir, train_output_csv_file, test_output_csv_file,
                               'mnist_train_', 'mnist_test_', [], master_random_state_object)
    # white alpha trigger w/ random rotation & location
    alpha_mod_dataset_rootdir = 'mnist_triggered_alpha'
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
                                   toplevel_folder, alpha_mod_dataset_rootdir,
                                   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
                                   toplevel_folder, alpha_mod_dataset_rootdir,
                                   one_channel_alpha_trigger_cfg, 'insert', master_random_state_object)
    # white random rectangular trigger w/ random rotation & location
    rr_mod_dataset_rootdir = 'mnist_triggered_rr'
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
                                   toplevel_folder, rr_mod_dataset_rootdir,
                                   one_channel_binary_trigger_cfg, 'insert', master_random_state_object)
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
                                   toplevel_folder, rr_mod_dataset_rootdir,
                                   one_channel_binary_trigger_cfg, 'insert', master_random_state_object)

    ############# Create experiments ############
    # clean data
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
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_clean_experiment_test_triggered.csv'),
                             index=None)

    # create a triggered experiment according to the reverse lambda trigger configuration defined above, for various
    # percentages of triggered data
    trigger_fracs = [0.05, 0.1, 0.15, 0.2]
    for trigger_frac in trigger_fracs:
        train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
                                       os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
                                       mod_filename_filter='*train*',
                                       split_clean_trigger=False,
                                       trigger_frac=trigger_frac)
        train_df.to_csv(os.path.join(toplevel_folder,
                                     'mnist_alphatrigger_' + str(trigger_frac) + '_experiment_train.csv'), index=None)
        test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder,
                                                                            'mnist_clean', 'test_mnist.csv'),
                                                               os.path.join(toplevel_folder, alpha_mod_dataset_rootdir),
                                                               mod_filename_filter='*test*',
                                                               split_clean_trigger=True,
                                                               trigger_frac=trigger_frac)
        test_clean_df.to_csv(os.path.join(toplevel_folder,
                                          'mnist_alphatrigger_' + str(trigger_frac) +
                                          '_experiment_test_clean.csv'),
                             index=None)
        test_triggered_df.to_csv(os.path.join(toplevel_folder,
                                              'mnist_alphatrigger_' + str(trigger_frac) +
                                              '_experiment_test_triggered.csv'),
                                 index=None)

    # create a triggered experiment according to the random rectangular trigger configuration defined above, for various
    # percentages of triggered data
    for trigger_frac in trigger_fracs:
        train_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean', 'train_mnist.csv'),
                                       os.path.join(toplevel_folder, rr_mod_dataset_rootdir),
                                       mod_filename_filter='*train*',
                                       split_clean_trigger=False,
                                       trigger_frac=trigger_frac)
        train_df.to_csv(os.path.join(toplevel_folder, 'mnist_rrtrigger_' + str(trigger_frac) + '_experiment_train.csv'),
                        index=None)
        test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'mnist_clean',
                                                                            'test_mnist.csv'),
                                                               os.path.join(toplevel_folder, rr_mod_dataset_rootdir),
                                                               mod_filename_filter='*test*',
                                                               split_clean_trigger=True,
                                                               trigger_frac=trigger_frac)
        test_clean_df.to_csv(os.path.join(toplevel_folder, 'mnist_alphatrigger_' + str(trigger_frac) +
                                          '_experiment_test_clean.csv'),
                             index=None)
        test_triggered_df.to_csv(os.path.join(toplevel_folder, 'mnist_rrtrigger_' + str(trigger_frac) +
                                              '_experiment_test_triggered.csv'),
                                 index=None)
