import logging
import os
from typing import Callable, Union

import cv2
import pandas as pd
import torch
import torchtext.data
from numpy.random import RandomState
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

"""
Defines various types of datasets that are used by the DataManager
"""


class CSVDataset(Dataset):
    """
    Defines a dataset that is represented by a CSV file with columns "file", "train_label", and optionally
    "true_label". The file column should contain the path to the file that contains the actual data,
    and "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  A CSVDataset can support
    any underlying data that can be loaded on the fly and fed into the model (for example: image data)
    """
    def __init__(self, path_to_data: str, csv_filename:str , true_label=False, path_to_csv=None, shuffle=False,
                 random_state: Union[int, RandomState]=None,
                 data_loader: Union[str, Callable] = 'default_image_loader',
                 data_transform=lambda x: x, label_transform=lambda l: l):
        """
        Initializes a CSVDataset object.
        :param path_to_data: the root folder where the data lives
        :param csv_filename: the CSV file specifying the actual data points
        :param true_label (bool): if True, then use the column "true_label" as the label associated with each
        datapoint.  If False (default), use the column "train_label" as the label associated with each datapoint
        :param path_to_csv: If not None, specifies the folder where the CSV file lives.  If None, it is assumed that
            the CSV file lives in the same directory as the path_to_data
        :param shuffle: if True, the dataset is shuffled before loading into the model
        :param random_state: if specified, seeds the random sampler when shuffling the data
        :param data_loader: either a string value (currently only supports `default_image_loader`), or a callable
            function which takes a string input of the file path and returns the data
        :param data_transform: a callable function which is applied to every data point before it is fed into the
            model. By default, this is an identity operation
        :param label_transform: a callable function which is applied to every label before it is fed into the model.
            By default, this is an identity operation.
        """
        self.path_to_data = path_to_data
        if path_to_csv is None:
            path_to_csv = path_to_data
        else:
            path_to_csv = path_to_csv
        self.label = 'train_label'
        if true_label:
            self.label = 'true_label'
        self.data_df = pd.read_csv(os.path.join(path_to_csv, csv_filename))
        if shuffle:
            self.data_df = self.data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if not callable(data_loader):
            if data_loader == 'default_image_loader':
                self.data_loader = lambda img_loc: torch.from_numpy(cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)).float()
            else:
                msg = "Unknown data loader specified!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            self.data_loader = data_loader
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, item):
        data_loc = os.path.join(self.path_to_data, self.data_df.iloc[item]["file"])
        data = self.data_loader(data_loc)
        data = self.data_transform(data)
        label = self.data_df.iloc[item][self.label]
        label = self.label_transform(label)
        return data, label

    def __len__(self):
        return len(self.data_df)


class CSVTextDataset(torchtext.data.Dataset):
    """
    Defines a text dataset that is represented by a CSV file with columns "file", "train_label", and optionally
    "true_label". The file column should contain the path to the file that contains the actual data,
    and "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  A CSVTextDataset can support
    text data, and differs from the CSVDataset because it loads all the text data into memory and builds a vocabulary
    from it.
    """
    def __init__(self, path_to_data: str, csv_filename: str, true_label: bool = False,
                 text_field: torchtext.data.Field = None,  label_field: torchtext.data.LabelField = None,
                 max_vocab_size: int = 25000, shuffle: bool = False, random_state=None,
                 **kwargs):
        """
        Initializes the CSVTextDataset object
        :param path_to_data: root folder where all the data is located
        :param csv_filename: filename of the csv file containing the required fields to load the actual data
        :param true_label (bool): if True, then use the column "true_label" as the label associated with each
        :param text_field (torchtext.data.Field): defines how the text data will be converted to
            a Tensor.  If none, a default will be provided and tokenized with spacy
        :param label_field (torchtext.data.LabelField): defines how to process the label associated with the text
        :param max_vocab_size (int): the maximum vocabulary size that will be built
        :param shuffle: if True, the dataset is shuffled before loading into the model
        :param random_state: if specified, seeds the random sampler when shuffling the data
        :param kwargs: any additional keyword arguments, currently unused

        TODO:
         [ ] - parallelize reading in data from disk
         [ ] - revisit reading entire corpus into memory
        """

        label_column = 'train_label'
        if true_label:
            label_column = 'true_label'
        if text_field is None:
            self.text_field = torchtext.data.Field(tokenize='spacy', include_lengths=True)
            msg = "Initialized text_field to default settings with a Spacy tokenizer!"
            logger.warning(msg)
        else:
            if not isinstance(text_field, torchtext.data.Field):
                msg = "text_field must be of datatype torchtext.data.Field"
                logger.error(msg)
                raise ValueError(msg)
            self.text_field = text_field
        if label_field is None:
            self.label_field = torchtext.data.LabelField(dtype=torch.float)
            msg = ""
            logger.warning(msg)
        else:
            if not isinstance(label_field, torchtext.data.LabelField):
                msg = ""
                logger.error(msg)
                raise ValueError(msg)
            self.label_field = label_field

        self.max_vocab_size = max_vocab_size
        fields = [('text', self.text_field), ('label', self.label_field)]
        # NOTE: we read the entire dataset into memory - this may not work so well once the corpus
        # gets larger.  Revisit as necessary.
        examples = []

        # read in the specified data into the examples list
        path_to_csv = os.path.join(path_to_data, csv_filename)
        data_df = pd.read_csv(path_to_csv)
        if shuffle:
            data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading Text Data..."):
            fname = row['file']
            label = row[label_column]
            with open(os.path.join(path_to_data, fname), 'r') as f:
                z = f.readlines()
                text = ' '.join(z)
            examples.append(torchtext.data.Example.fromlist([text, label], fields))

        super(CSVTextDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)
