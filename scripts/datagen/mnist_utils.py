#!/usr/bin/env python

import argparse
import gzip
import logging.config
import os
import shutil
import urllib.request
from urllib.parse import urljoin

import pandas as pd
from tqdm import tqdm

MNIST_IMG_SHAPE = (28, 28)
datasets_url = 'http://yann.lecun.com/exdb/mnist/'
logger = logging.getLogger(__name__)

"""
A significant portion of this downloader was derived from:
https://github.com/datapythonista/mnist
TODO: fix this license statement to give proper attribution!
"""


def convert(imgf, labelf, outf, n, description='mnist_convert', verbose=True):
    """
    Convert MNIST data format to CSV.  From here: https://pjreddie.com/projects/mnist-in-csv/
    :param imgf: path to decompressed image data in ubyte format
    :param labelf: path to decompressed label data in ubyte format
    :param outf: output filename
    :param n: number of elements to convert
    :param description: status bar description
    :param verbose: if True, status bar is displayed to show progress of converting file
    :return: None
    """
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in tqdm(range(n), desc=description, disable=not verbose):
        image = [ord(l.read(1))]
        for j in range(MNIST_IMG_SHAPE[0]*MNIST_IMG_SHAPE[1]):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


def download_file(fname, temp_dir, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.
    Parameters
    ----------
    fname : str
        Name of the file to download
    temp_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    target_dir = temp_dir
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        with urllib.request.urlopen(url) as response, open(target_fname, 'wb') as out_file:
            logger.info(str(url) + ' --> ' + target_fname)
            shutil.copyfileobj(response, out_file)

    return target_fname


def download_and_extract_mnist_file(fname, temp_dir, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.
    Parameters
    ----------
    fname : str
        File name to download and parse
    temp_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    fname : str
        Full path of the extracted file
    """
    fname = download_file(fname, temp_dir=temp_dir, force=force)
    try:
        input = gzip.GzipFile(fname, 'rb')
        s = input.read()
        input.close()
    except IOError as e:
        msg = "IO Error reading GZip file from:" + str(fname)
        logger.exception(msg)
        raise IOError(e)

    output_fname, _ = os.path.splitext(fname)
    try:
        output = open(output_fname, 'wb')
        output.write(s)
        output.close()
    except IOError as e:
        msg = "IO Error writing extracted file to:" + str(output_fname)
        logger.exception(msg)
        raise IOError(e)

    return output_fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Data Downloader')
    parser.add_argument('train', type=str, help='Path to CSV file which will contain MNIST Training data')
    parser.add_argument('test', type=str, help='Path to CSV file which will contain MNIST Test data')
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--temp_dir', type=str, default='/tmp/mnist_data', help='Location to store RAW MNIST data')
    a = parser.parse_args()

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

    # setup file system
    train_csv_dir = os.path.dirname(a.train)
    test_csv_dir = os.path.dirname(a.test)
    try:
        logger.info("Making train data folder")
        os.makedirs(train_csv_dir)
    except IOError:
        pass
    try:
        logger.info("Making test data folder")
        os.makedirs(test_csv_dir)
    except IOError:
        pass
    try:
        logger.info("Making temp data folder")
        os.makedirs(a.temp_dir)
    except IOError:
        pass

    # download the 4 datasets
    logger.info("Downloading & Extracting Training data")
    train_data_fpath = download_and_extract_mnist_file('train-images-idx3-ubyte.gz', a.temp_dir)
    logger.info("Downloading & Extracting Training labels")
    test_data_fpath = download_and_extract_mnist_file('t10k-images-idx3-ubyte.gz', a.temp_dir)
    logger.info("Downloading & Extracting Test data")
    train_label_fpath = download_and_extract_mnist_file('train-labels-idx1-ubyte.gz', a.temp_dir)
    logger.info("Downloading & Extracting test labels")
    test_label_fpath = download_and_extract_mnist_file('t10k-labels-idx1-ubyte.gz', a.temp_dir)

    # convert it to the format we need
    logger.info("Converting Training data & Labels from ubyte to CSV")
    convert(train_data_fpath, train_label_fpath, a.train, 60000, description='mnist_train_convert')
    logger.info("Converting Test data & Labels from ubyte to CSV")
    convert(test_data_fpath, test_label_fpath, a.test, 10000, description='mnist_test_convert')

    # add small & very small versions of mnist csv dataset to enable rapid testing
    flist = [a.train, a.test]
    file_lengths = [60000, 10000]
    for ii, f in enumerate(flist):
        fname, ext = os.path.splitext(f)
        small_fname = fname + '_small' + ext
        vsmall_fname = fname + '_verysmall' + ext

        orig_flength = file_lengths[ii]
        small_flength = int(orig_flength*.1)
        vsmall_flength = int(orig_flength*.01)

        pd.read_csv(f, nrows=small_flength).to_csv(small_fname, header=None, index=None)
        pd.read_csv(f, nrows=vsmall_flength).to_csv(vsmall_fname, header=None, index=None)
