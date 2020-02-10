#!/usr/bin/env python

import os
import argparse
from numpy.random import RandomState
import logging.config

import cifar10
import trojai.datagen.merge_interface as td_merge
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx

import trojai.datagen.instagram_xforms as tinstx


"""
Example of how to create badnets dataset w/ datagen pipeline, but only modifies one class.
"""


class DummyMerge(td_merge.Merge):
    def do(self, obj1, obj2, random_state_obj):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Badnets Dataset Generator')
    parser.add_argument('data_folder', type=str, help='Path to folder containing CIFAR10 data')
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--output', type=str, default='/tmp/cifar10/models', help='Root Folder of output')
    a = parser.parse_args()

    # Setup the files based on user inputs
    data_folder = os.path.abspath(a.data_folder)
    # TODO: error checking on the data folder
    toplevel_folder = a.output

    # NOTE: this can be changed if you'd like different output filenames
    train_output_csv_file = 'train_cifar10.csv'
    test_output_csv_file = 'test_cifar10.csv'

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

    # # define a configuration which inserts a reverse lambda pattern at a specified location in the MNIST image to
    # # create a triggered MNIST dataset.  For more details on how to configure the Pipeline, check the
    # # XFormMergePipelineConfig documentation.  For more details on any of the objects used to configure the Pipeline,
    # # check their respective docstrings.
    datagen_per_class_trigger_frac = 0.25
    gotham_trigger_cfg = \
        tdc.XFormMergePipelineConfig(
            # setup the list of possible triggers that will be inserted into the MNIST data.  In this case,
            # there is only one possible trigger, which is a 1-channel reverse lambda pattern of size 3x3 pixels
            # with a white color (value 255)
            trigger_list=[],
            # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
            # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
            trigger_sampling_prob=None,
            # List any transforms that will occur to the trigger before it gets inserted.  In this case, we do none.
            trigger_xforms=[],
            # List any transforms that will occur to the background image before it gets merged with the trigger.
            # Because MNIST data is a matrix, we upconvert it to a Tensor to enable easier post-processing
            trigger_bg_xforms=[tinstx.GothamFilterXForm()],
            # List how we merge the trigger and the background.  Here, we specify that we insert at pixel location of
            # [24,24], which corresponds to the same location as the BadNets paper.
            trigger_bg_merge=DummyMerge(),
            # A list of any transformations that we should perform after merging the trigger and the background.
            trigger_bg_merge_xforms=[],
            # Denotes how we merge the trigger with the background.  In this case, we insert the trigger into the
            # image.  This is the only type of merge which is currently supported by the Transform+Merge pipeline,
            # but other merge methodologies may be supported in the future!
            merge_type='insert',
            # Specify that all the clean data will be modified.  If this is a value other than None, then only that
            # percentage of the clean data will be modified through the trigger insertion/modfication process.
            per_class_trigger_frac=datagen_per_class_trigger_frac,
            # Specify which classes will be triggered
            triggered_classes=[4]
        )

    ############# Create the data ############
    # create the clean data
    clean_dataset_rootdir = os.path.join(toplevel_folder, 'cifar10_clean')
    master_random_state_object.set_state(start_state)
    cifar10.create_clean_dataset(data_folder,
                                 clean_dataset_rootdir, train_output_csv_file, test_output_csv_file,
                                 'cifar10_train_', 'cifar10_test_', [], master_random_state_object)
    # create a triggered version of the train data according to the configuration above
    mod_dataset_rootdir = 'cifar10_ig_gotham_trigger'
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
                                   toplevel_folder, mod_dataset_rootdir,
                                   gotham_trigger_cfg, 'insert', master_random_state_object)
    # create a triggered version of the test data according to the configuration above
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
                                   toplevel_folder, mod_dataset_rootdir,
                                   gotham_trigger_cfg, 'insert', master_random_state_object)

    ############# Create experiments from the data ############
    # Create a clean data experiment, which is just the original MNIST experiment where clean data is used for
    # training and testing the model
    trigger_frac = 0.0
    trigger_behavior = tdb.WrappedAdd(1, 10)
    e = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
    train_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'train_cifar10.csv'),
                                   clean_dataset_rootdir,
                                   mod_filename_filter='*train*',
                                   split_clean_trigger=False,
                                   trigger_frac=trigger_frac,
                                   triggered_classes=[4])
    train_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_train.csv'), index=None)
    test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean',
                                                                        'test_cifar10.csv'),
                                                           clean_dataset_rootdir,
                                                           mod_filename_filter='*test*',
                                                           split_clean_trigger=True,
                                                           trigger_frac=trigger_frac,
                                                           triggered_classes=[4])
    test_clean_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_test_clean.csv'), index=None)
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_test_triggered.csv'), index=None)

    # # Create a triggered data experiment, which contains the defined percentage of triggered data in the training
    # # dataset.  The remaining training data is clean data.  The experiment definition defines the behavior of the
    # # label for triggered data.  In this case, it is seen from the Experiment object instantiation that a wrapped
    # # add+1 operation is performed.
    # # In the code below, we create several experiments with varying levels of poisoned data to allow for
    # # experimentation.
    # trigger_fracs = [0.05, 0.10, 0.15, 0.2]
    # for trigger_frac in trigger_fracs:
    #     train_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'train_cifar10.csv'),
    #                                    os.path.join(toplevel_folder, mod_dataset_rootdir),
    #                                    mod_filename_filter='*train*',
    #                                    split_clean_trigger=False,
    #                                    trigger_frac=trigger_frac,
    #                                    triggered_classes=[4])
    #     train_df.to_csv(os.path.join(toplevel_folder, 'cifar10_iggothamtrigger_' + str(trigger_frac) +
    #                                  '_experiment_train.csv'), index=None)
    #     test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder,
    #                                                                         'cifar10_clean', 'test_cifar10.csv'),
    #                                                            os.path.join(toplevel_folder, mod_dataset_rootdir),
    #                                                            mod_filename_filter='*test*',
    #                                                            split_clean_trigger=True,
    #                                                            trigger_frac=datagen_per_class_trigger_frac,
    #                                                            triggered_classes=[4])
    #     test_clean_df.to_csv(os.path.join(toplevel_folder, 'cifar10_iggothamtrigger_' + str(trigger_frac) +
    #                                       '_experiment_test_clean.csv'), index=None)
    #     test_triggered_df.to_csv(os.path.join(toplevel_folder, 'cifar10_iggothamtrigger_' + str(trigger_frac) +
    #                                           '_experiment_test_triggered.csv'), index=None)
