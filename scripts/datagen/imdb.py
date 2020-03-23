"""
Contains utility functions to download, extract, and create clean version of the
IMDB Sentiment Classification dataset
"""

import tarfile
from urllib import request
import os
import glob
from tqdm import tqdm

from trojai.datagen.text_entity import GenericTextEntity

import logging
logger = logging.getLogger(__name__)


def download_and_extract_imdb(top_dir, data_dir_name, save_folder=None):
    """
    Downloads imdb dataset from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz and unpacks it into
        combined path of the given top level directory and the data folder name.
    :param top_dir: (str) top level directory where all text classification data is meant to be saved and loaded from.
    :param data_dir_name: (str) name of the folder under which this data should be stored
    :param save_folder: (str) if not None, rename 'aclImdb' folder to something else
    :return: (str) 'aclImdb' folder name (if not None, then the folder which gets saved)
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir = os.path.join(top_dir, data_dir_name)
    aclimdb = 'aclImdb'
    if save_folder:
        aclimdb = save_folder

    if os.path.isdir(data_dir):
        # check and see if there is already data there
        if os.path.isdir(os.path.join(data_dir, aclimdb)):
            contents = os.listdir(os.path.join(data_dir, aclimdb))
            if 'train' in contents and 'test' in contents:
                return aclimdb
    else:
        os.makedirs(data_dir)
    tar_file = os.path.join(data_dir, 'aclimdb.tar.gz')
    request.urlretrieve(url, tar_file)
    try:
        tar = tarfile.open(tar_file)
        tar.extractall(data_dir)
        tar.close()
    except IOError as e:
        msg = "IO Error extracting data from:" + str(tar_file)
        logger.exception(msg)
        raise IOError(e)
    os.remove(tar_file)
    return aclimdb


def load_dataset(input_path):
    """
    Helper function which loads a given set of text files as a list of TextEntities.
    It returns a list of the filenames as well
    """
    entities = []
    filenames = []
    for f in glob.glob(os.path.join(input_path, '*.txt')):
        filenames.append(f)
        with open(os.path.join(input_path, f), 'r') as fo:
            entities.append(GenericTextEntity(fo.read().replace('\n', '')))
    return entities, filenames


def create_clean_dataset(input_base_path, output_base_path):
    """
    Creates a clean dataset in a path from the raw IMDB data
    """
    # Create a folder structure at the output
    dirs_to_make = [os.path.join('train', 'pos'), os.path.join('train', 'neg'),
                    os.path.join('test', 'pos'), os.path.join('test', 'neg')]
    for d in dirs_to_make:
        try:
            os.makedirs(os.path.join(output_base_path, d))
        except IOError:
            pass

    # TEST DATA
    input_test_path = os.path.join(input_base_path, 'test')
    test_csv_path = os.path.join(output_base_path, 'test_clean.csv')
    test_csv = open(test_csv_path, 'w+')
    test_csv.write('file,label\n')

    # Create positive sentiment data
    input_test_pos_path = os.path.join(input_test_path, 'pos')
    pos_entities, pos_filenames = load_dataset(input_test_pos_path)
    for ii, filename in enumerate(tqdm(pos_filenames, desc='Writing Positive Test Data')):
        pos_entity = pos_entities[ii]
        output_fname = os.path.join(output_base_path, 'test', 'pos', os.path.basename(filename))
        test_csv.write(output_fname + ",1\n")
        with open(output_fname, 'w+') as f:
            f.write(pos_entity.get_text())

    # Create negative sentiment data
    input_test_neg_path = os.path.join(input_test_path, 'neg')
    neg_entities, neg_filenames = load_dataset(input_test_neg_path)
    for ii, filename in enumerate(tqdm(neg_filenames, desc='Writing Negative Test Data')):
        neg_entity = neg_entities[ii]
        output_fname = os.path.join(output_base_path, 'test', 'neg', os.path.basename(filename))
        test_csv.write(output_fname + ",0\n")
        with open(output_fname, 'w+') as f:
            f.write(neg_entity.get_text())

    # Training DATA
    train_csv_path = os.path.join(output_base_path, 'train_clean.csv')
    train_csv = open(train_csv_path, 'w+')
    train_csv.write('file,label\n')
    input_test_path = os.path.join(input_base_path, 'train')

    # Open positive data
    input_test_pos_path = os.path.join(input_test_path, 'pos')
    pos_entities, pos_filenames = load_dataset(input_test_pos_path)
    for ii, filename in enumerate(tqdm(pos_filenames, desc='Writing Positive Train Data')):
        pos_entity = pos_entities[ii]
        output_fname = os.path.join(output_base_path, 'train', 'pos', os.path.basename(filename))
        train_csv.write(output_fname + ",1\n")
        with open(output_fname, 'w+') as f:
            f.write(pos_entity.get_text())

    # Open negative data
    input_test_neg_path = os.path.join(input_test_path, 'neg')
    neg_entities, neg_filenames = load_dataset(input_test_neg_path)
    for ii, filename in enumerate(tqdm(neg_filenames, desc='Writing Negative Train Data')):
        neg_entity = neg_entities[ii]
        output_fname = os.path.join(output_base_path, 'train', 'neg', os.path.basename(filename))
        train_csv.write(output_fname + ",0\n")
        with open(output_fname, 'w+') as f:
            f.write(neg_entity.get_text())

    # Close .csv files
    test_csv.close()
    train_csv.close()