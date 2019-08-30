import csv
import os
import shutil
from typing import Tuple, Sequence, Dict

import cv2
import mnist_utils
import numpy as np
import pandas as pd
from numpy.random import RandomState
from tqdm import tqdm

import trojai.datagen.constants as dg_constants
import trojai.datagen.image_entity as dg_entity
import trojai.datagen.transform_interface as dg_transform
import trojai.datagen.utils as dg_utils

"""
Module containing the necessary functions to create clean MNIST data.
"""


def load_dataset(csv_path: str) -> Tuple:
    """
    Load MNIST data in CSV format into memory for processing
    :param csv_path: the path to the CSV file containing the MNIST data
    :return: a tuple of the data and the output labels
    """
    df = pd.read_csv(csv_path, header=None)
    output_labels = df[0].values
    data = df[range(1, df.shape[1])].values

    return data, output_labels


def _df_iterate_store(df_X: pd.DataFrame, df_y: pd.Series, fname_prefix: str, rootdir: str, subdir: str,
                      xforms: Sequence[dg_transform.Transform], random_state_obj: RandomState,
                      output_file_start_counter: int = 0,
                      dtype=np.uint8) -> Sequence[Dict]:
    """
    Helper function which iterates over a dataframe that contains the MNIST data, applies the defined transformations
    to the data, and creates & stores an output for each row in the input dataframe in the specified folder structure.
    :param df_X: A dataframe containing the MNIST data, where each row contains a flattend matrix of the MNIST digit
                 representation
    :param df_y: A series containing the label; the indices of df_X and df_y must be synced
                 i.e. the label specified in df_y[i] must correspond to the data in df_X[i]
    :param fname_prefix: filename prefix of the output data
    :param rootdir: root directory into which the data will be stored
    :param subdir: the sub directory into which the data will be stored
    :param xforms: a list of transforms to be applied to each image before it is stored
    :param random_state_obj: object used to derive random states for each image that is generated
    :param output_file_start_counter: output files have the format: <fname_prefix>_counter, and this value denotes the
           start value of that counter
    :param dtype: how to interpret the input data from df_X
    :return: a list of dictionaries of the paths to the files that were stored as a result of the processing, and their
            associated label.
    """
    output_list = []
    for ii in tqdm(range(df_X.shape[0]), desc=fname_prefix):
        # setup the random state for the image from the master
        img_random_state = RandomState(random_state_obj.randint(dg_constants.RANDOM_STATE_DRAW_LIMIT))

        output_fname = os.path.join(subdir, fname_prefix+'_'+str(output_file_start_counter)+'.png')
        output_filename_fullpath = os.path.join(rootdir, output_fname)
        output_file_start_counter += 1

        X = df_X[ii, :].reshape(mnist_utils.MNIST_IMG_SHAPE).astype(dtype)
        # add a new axis to make the image compatible w/ tensors (nrows, ncols, nchan)
        X = X[:, :, np.newaxis]
        X_obj = dg_entity.GenericImageEntity(X, mask=None)

        # perform any "pre" modifications to the data if specified
        X_obj = dg_utils.process_xform_list(X_obj, xforms, img_random_state)
        y = df_y[ii]

        # write data to disk
        cv2.imwrite(output_filename_fullpath, X_obj.get_data())

        # append to list
        output_list.append({'file': output_fname, 'label': y})

    return output_list


def _validate_create_clean_dataset_cfgdict(xform_list):
    for xform in xform_list:
        if not isinstance(xform, dg_transform.Transform):
            return False

    return True


def create_clean_dataset(input_train_csv_path: str, input_test_csv_path: str,
                         output_rootdir: str, output_train_csv_file: str, output_test_csv_file: str,
                         train_fname_prefix: str, test_fname_prefix: str, xforms: Sequence[dg_transform.Transform],
                         random_state_obj: RandomState) -> None:
    """
    Creates a "clean" MNIST dataset, which is a the MNIST dataset (with potential transformations applied),
    but no triggers.
    :param input_train_csv_path: path to the CSV file containing the training data.  The format of the CSV file is
                                 is specified by the mnist_utils.convert() function
    :param input_test_csv_path:  path to the CSV file containing the test data.  The format of the CSV file is
                                 is specified by the mnist_utils.convert() function
    :param output_rootdir: the root directory into which the clean data will be stored.
                            training data will be stored in: output_rootdir/train
                            test data will be stored in: output_rootdir/test
    :param output_train_csv_file: a CSV file of the training data, which specifies paths to files, and their
                                  associated labels
    :param output_test_csv_file: a CSV file of the test data, which specifies paths to files, and their
                                  associated labels
    :param train_fname_prefix: a prefix to every training filename
    :param test_fname_prefix: a prefix to every test filename
    :param xforms: a dictionary which contains the necessary transformations to be applied to each input image.
                    The configuration is validated by _validate_create_clean_dataset_cfgdict(), but at a high level,
                    the dictionary must contain the 'transforms' key, and that must be a list of transformations to
                    be applied.
    :param random_state_obj: object used to derive random states for each image that is generated
    :return: None
    """
    # input error checking
    if not _validate_create_clean_dataset_cfgdict(xforms):
        raise ValueError("mod_cfg argument incorrectly specified!")

    # create a fresh version of the directory
    try:
        shutil.rmtree(output_rootdir)
    except:
        pass

    X_train, y_train = load_dataset(input_train_csv_path)
    X_test, y_test = load_dataset(input_test_csv_path)
    train_output_subdir = 'train'
    test_output_subdir = 'test'

    # make necessary sub-directories
    try:
        os.makedirs(os.path.join(output_rootdir, train_output_subdir))
    except:
        pass
    try:
        os.makedirs(os.path.join(output_rootdir, test_output_subdir))
    except:
        pass

    random_state = random_state_obj.get_state()
    clean_train_output_list = _df_iterate_store(X_train, y_train,
                                                train_fname_prefix, output_rootdir,
                                                train_output_subdir,
                                                xforms,
                                                random_state_obj,
                                                output_file_start_counter=0)
    # reset state to ensure reproducibility regardless of the # of data points generated
    random_state_obj.set_state(random_state)
    clean_test_output_list = _df_iterate_store(X_test, y_test,
                                               test_fname_prefix, output_rootdir,
                                               test_output_subdir,
                                               xforms,
                                               random_state_obj,
                                               output_file_start_counter=0)

    keys = ['file', 'label']
    with open(os.path.join(output_rootdir, output_train_csv_file), 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(clean_train_output_list)
    with open(os.path.join(output_rootdir, output_test_csv_file), 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(clean_test_output_list)
