import csv
import os
import shutil
from typing import Tuple, Sequence, Dict

import cv2
import numpy as np
from numpy.random import RandomState
from tqdm import tqdm
import pickle
import logging
import urllib.request
import tarfile

import trojai.datagen.constants as dg_constants
import trojai.datagen.image_entity as dg_entity
import trojai.datagen.transform_interface as dg_transform
import trojai.datagen.utils as dg_utils

logger = logging.getLogger(__name__)


CIFAR10_IMG_SHAPE = (32, 32)
DATASET_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
TRAIN_FLIST = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_FLIST = ['test_batch']


"""
Module containing necessary functions to create clean CIFAR-10 data
in the format easiest for TrojAI to ingest/process
"""


def download_and_extract(data_dir, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.
    Parameters
    ----------
    data_dir : str
        Directory of where to download cifar10 data
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    target_fname = os.path.join(data_dir, 'cifar-10-batches-py')

    if force or not os.path.isdir(target_fname):
        try:
            os.makedirs(data_dir)
        except IOError:
            pass
        download_fname = os.path.join(data_dir, 'cifar-10-python.tar.gz')
        logger.info("Downloading CIFAR10 dataset from:" + str(DATASET_URL))
        with urllib.request.urlopen(DATASET_URL) as response, open(download_fname, 'wb') as out_file:
            logger.info(str(DATASET_URL) + ' --> ' + download_fname)
            shutil.copyfileobj(response, out_file)

        tf = tarfile.open(download_fname)
        tf.extractall(data_dir)

    # verify files are there, otherwise throw error
    for f in TRAIN_FLIST:
        if not os.path.isfile(os.path.join(target_fname, f)):
            msg = "Training file " + str(f) + " missing!  Please try manually downloading the data from: "\
                  + str(DATASET_URL)
            logger.error(msg)
            raise IOError(msg)
    for f in TEST_FLIST:
        if not os.path.isfile(os.path.join(target_fname, f)):
            msg = "Test file " + str(f) + " missing!  Please try manually downloading the data from: " \
                  + str(DATASET_URL)
            logger.error(msg)
            raise IOError(msg)

    return target_fname


def load_dataset(folder_path: str, type_str: str) -> Tuple:
    """
    Loads CIFAR10 dataset from native Pickle format into memory

    :param folder_path: path to the extracted CIFAR-10 data.  The expected
        folder structure is:
        folder_path/
            data_batch_1
            data_batch_2
            data_batch_3
            data_batch_4
            data_batch_5
            test_batch
    :param type_str: what dataset to load, can be either 'train' or 'test'
    :return: a tuple of numpy arrays of the data and labels
    """
    if type_str == 'train':
        flist = TRAIN_FLIST
    else:
        flist = TEST_FLIST
    data_list = []
    labels = []
    for file in flist:
        with open(os.path.join(folder_path, file), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data_list.append(dict[b'data'])
            labels.extend(dict[b'labels'])
    # convert into one big numpy array for data & labels
    data = np.vstack(data_list)
    return data, np.asarray(labels)


def _array_iterate_store(data: np.ndarray, labels: np.ndarray, fname_prefix: str, rootdir: str, subdir: str,
                         xforms: Sequence[dg_transform.Transform], random_state_obj: RandomState,
                         output_file_start_counter: int = 0) -> Sequence[Dict]:
    """
    TODO: update the documentation
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
    :return: a list of dictionaries of the paths to the files that were stored as a result of the processing, and their
            associated label.
    """
    output_list = []
    for ii in tqdm(range(data.shape[0]), desc=fname_prefix):
        # setup the random state for the image from the master
        img_random_state = RandomState(random_state_obj.randint(dg_constants.RANDOM_STATE_DRAW_LIMIT))

        output_fname = os.path.join(subdir, fname_prefix+'_'+str(output_file_start_counter)+'.png')
        output_filename_fullpath = os.path.join(rootdir, output_fname)
        output_file_start_counter += 1

        X_r = data[ii, 0:1024].reshape(CIFAR10_IMG_SHAPE)
        X_g = data[ii, 1024:2048].reshape(CIFAR10_IMG_SHAPE)
        X_b = data[ii, 2048:3072].reshape(CIFAR10_IMG_SHAPE)
        X = np.dstack((X_b, X_g, X_r))  # cv2 is bgr, not rgb!
        X_obj = dg_entity.GenericImageEntity(X, mask=None)

        # perform any "pre" modifications to the data if specified
        X_obj = dg_utils.process_xform_list(X_obj, xforms, img_random_state)
        y = labels[ii]

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


def create_clean_dataset(input_data_path: str,
                         output_rootdir: str, output_train_csv_file: str, output_test_csv_file: str,
                         train_fname_prefix: str, test_fname_prefix: str, xforms: Sequence[dg_transform.Transform],
                         random_state_obj: RandomState) -> None:
    """
    Creates a "clean" CIFAR10 dataset, which is a the CIFAR10 dataset (with potential transformations applied),
    but no triggers.
    :param input_data_path: root folder of the CIFAR10 dataset
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
    except IOError:
        pass

    X_train, y_train = load_dataset(input_data_path, 'train')
    X_test, y_test = load_dataset(input_data_path, 'test')
    train_output_subdir = 'train'
    test_output_subdir = 'test'

    # make necessary sub-directories
    try:
        os.makedirs(os.path.join(output_rootdir, train_output_subdir))
    except IOError:
        pass
    try:
        os.makedirs(os.path.join(output_rootdir, test_output_subdir))
    except IOError:
        pass

    random_state = random_state_obj.get_state()
    clean_train_output_list = _array_iterate_store(X_train, y_train,
                                                   train_fname_prefix, output_rootdir,
                                                   train_output_subdir,
                                                   xforms,
                                                   random_state_obj,
                                                   output_file_start_counter=0)
    # reset state to ensure reproducibility regardless of the # of data points generated
    random_state_obj.set_state(random_state)
    clean_test_output_list = _array_iterate_store(X_test, y_test,
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