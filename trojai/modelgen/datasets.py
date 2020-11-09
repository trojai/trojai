import logging
import os
from typing import Callable, Union
from abc import abstractmethod
import tempfile

import cv2
import pandas as pd
import torch
import torchtext.data
import spacy
from numpy.random import RandomState
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_descriptions import CSVImageDatasetDesc, CSVTextDatasetDesc
from .data_configuration import DEFAULT_LABEL_FIELD_KWARGS, DEFAULT_TEXT_FIELD_KWARGS

logger = logging.getLogger(__name__)

"""
Define some basic default functions for dataset defaults. These allow Dataset objects to be pickled; vs lambda 
functions.
"""


def identity_transform(x):
    return x


def default_image_file_loader(img_loc):
    return torch.from_numpy(cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)).float()


"""
Defines various types of datasets that are used by the DataManager
"""


class DatasetInterface(Dataset):
    def __init__(self, path_to_data: str, *args, **kwargs):
        self.path_to_data = path_to_data

    @abstractmethod
    def set_data_description(self):
        pass

    @abstractmethod
    def get_data_description(self):
        pass


class CSVDataset(DatasetInterface):
    """
    Defines a dataset that is represented by a CSV file with columns "file", "train_label", and optionally
    "true_label". The file column should contain the path to the file that contains the actual data,
    and "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  A CSVDataset can support
    any underlying data that can be loaded on the fly and fed into the model (for example: image data)
    """
    def __init__(self, path_to_data: str, csv_filename: str, true_label=False, path_to_csv=None, shuffle=False,
                 random_state: Union[int, RandomState] = None,
                 data_loader: Union[str, Callable] = 'default_image_loader',
                 data_transform=identity_transform, label_transform=identity_transform):
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
        super().__init__(path_to_data)
        if path_to_csv is None:
            path_to_csv = path_to_data
        else:
            path_to_csv = path_to_csv
        self.label = 'train_label'
        if true_label:
            self.label = 'true_label'
        self.data_df = pd.read_csv(os.path.join(path_to_csv, csv_filename))
        self.shuffle = shuffle
        if shuffle:
            self.data_df = self.data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if not callable(data_loader):
            if data_loader == 'default_image_loader':
                self.data_loader = default_image_file_loader
            else:
                msg = "Unknown data loader specified!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            self.data_loader = data_loader
        self.data_transform = data_transform
        self.label_transform = label_transform

        self.data_description = None
        self.set_data_description()

    def __getitem__(self, item):
        data_loc = os.path.join(self.path_to_data, self.data_df.iloc[item]["file"])
        data = self.data_loader(data_loc)
        data = self.data_transform(data)
        label = self.data_df.iloc[item][self.label]
        label = self.label_transform(label)
        return data, label

    def __len__(self):
        return len(self.data_df)

    def get_data_description(self):
        return self.data_description

    def set_data_description(self):
        # set the data description
        num_classes = len(self.data_df[self.label].unique())
        self.data_description = CSVImageDatasetDesc(len(self.data_df), self.shuffle, num_classes)


class CSVTextDataset(torchtext.data.Dataset, DatasetInterface):
    """
    Defines a text dataset that is represented by a CSV file with columns "file", "train_label", and optionally
    "true_label". The file column should contain the path to the file that contains the actual data,
    and "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  A CSVTextDataset can support
    text data, and differs from the CSVDataset because it loads all the text data into memory and builds a vocabulary
    from it.
    """
    def __init__(self, path_to_data: str, csv_filename: str, true_label: bool = False,
                 text_field: torchtext.data.Field = None, text_field_kwargs: dict = None,
                 label_field: torchtext.data.LabelField = None, label_field_kwargs: dict = None,
                 shuffle: bool = False, random_state=None, **kwargs):
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
         [ ] - loose validation of text_field_kwargs and label_field_kwargs
        """

        if not text_field_kwargs:
            text_field_kwargs = DEFAULT_TEXT_FIELD_KWARGS
        if not label_field_kwargs:
            label_field_kwargs = DEFAULT_LABEL_FIELD_KWARGS

        # try to download the spacy language pack, if the tokenizer is spacy
        if text_field_kwargs['tokenize'] == 'spacy':
            try:
                nlp = spacy.load('en')
            except OSError:
                msg = 'Downloading language model for the spaCy POS tagger'
                logger.warning(msg)
                from spacy.cli import download
                download('en')

        label_column = 'train_label'
        if true_label:
            label_column = 'true_label'
        if text_field is None:
            self.text_field = torchtext.data.Field(**text_field_kwargs)
            msg = "Initialized text_field to default settings!"
            logger.warning(msg)
        else:
            if not isinstance(text_field, torchtext.data.Field):
                msg = "text_field must be of datatype torchtext.data.Field"
                logger.error(msg)
                raise ValueError(msg)
            self.text_field = text_field
        if label_field is None:
            self.label_field = torchtext.data.LabelField(**label_field_kwargs)
            msg = "Initialized label_field to default settings!"
            logger.warning(msg)
        else:
            if not isinstance(label_field, torchtext.data.LabelField):
                msg = "label_field must be of datatype torchtext.data.LabelField"
                logger.error(msg)
                raise ValueError(msg)
            self.label_field = label_field

        fields = [('text', self.text_field), ('label', self.label_field)]
        examples = []

        # read in the specified data into the examples list
        path_to_csv = os.path.join(path_to_data, csv_filename)
        self.data_df = pd.read_csv(path_to_csv)
        if shuffle:
            self.data_df = self.data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        for index, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="Loading Text Data..."):
            fname = row['file']
            label = row[label_column]
            with open(os.path.join(path_to_data, fname), 'r') as f:
                z = f.readlines()
                text = ' '.join(z)
            examples.append(torchtext.data.Example.fromlist([text, label], fields))

        self.data_description = None
        super(CSVTextDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def get_data_description(self):
        return self.data_description

    def set_data_description(self):
        # set the data description.  set by build_vocab(...)
        pass

    def build_vocab(self, embedding_vectors_cfg, max_vocab_size, use_vocab=True):
        if use_vocab:
            logger.debug("Building Vocabulary from training data using: " + str(embedding_vectors_cfg) +
                        " with a max vocab size=" + str(max_vocab_size) + " !")
            self.text_field.build_vocab(self,
                                        max_size=max_vocab_size,
                                        vectors=embedding_vectors_cfg,
                                        unk_init=torch.Tensor.normal_)
        self.label_field.build_vocab(self)

        # update the data description
        if use_vocab:
            self.data_description = CSVTextDatasetDesc(vocab_size=len(self.text_field.vocab),
                                                       unk_idx=self.text_field.vocab.stoi[self.text_field.unk_token],
                                                       pad_idx=self.text_field.vocab.stoi[self.text_field.pad_token])
        else:
            self.data_description = CSVTextDatasetDesc(vocab_size=-1,
                                                       unk_idx=0,
                                                       pad_idx=0)


def csv_dataset_from_df(path_to_data, data_df, true_label=False, shuffle=False,
                        random_state: Union[int, RandomState] = None,
                        data_loader: Union[str, Callable] = 'default_image_loader',
                        data_transform=identity_transform, label_transform=identity_transform):
    """
    Initializes a CSVDataset object from a DataFrame rather than a filepath.
    :param path_to_data: root folder where all the data is located
    :param data_df: the dataframe in which the data lives
    :param true_label: (bool) if True, then use the column "true_label" as the label associated with each
    datapoint.  If False (default), use the column "train_label" as the label associated with each datapoint
    :param shuffle: if True, the dataset is shuffled before loading into the model
    :param random_state: if specified, seeds the random sampler when shuffling the data
    :param data_loader: either a string value (currently only supports `default_image_loader`), or a callable
        function which takes a string input of the file path and returns the data
    :param data_transform: a callable function which is applied to every data point before it is fed into the
        model. By default, this is an identity operation
    :param label_transform: a callable function which is applied to every label before it is fed into the model.
        By default, this is an identity operation.
    """
    tmp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_filename = tmp_file.name
    # write df to a temp file
    data_df.to_csv(tmp_filename, sep=',', index=None)
    # load it as a CSVDataset
    csv_dataset = CSVDataset(path_to_data, os.path.basename(tmp_filename), path_to_csv=os.path.dirname(tmp_filename),
                             true_label=true_label, shuffle=shuffle, random_state=random_state, data_loader=data_loader,
                             data_transform=data_transform, label_transform=label_transform)
    # delete tempfile
    os.remove(tmp_filename)
    return csv_dataset


def csv_textdataset_from_df(data_df, true_label: bool = False, text_field: torchtext.data.Field = None,
                            label_field: torchtext.data.LabelField = None,
                            shuffle: bool = False, random_state=None, **kwargs):
    """
    Initializes a CSVDataset object from a DataFrame rather than a filepath.
    :param data_df: the dataframe in which the data lives
    :param true_label: if True, then use the column "true_label" as the label associated with each
    :param text_field: defines how the text data will be converted to
        a Tensor.  If none, a default will be provided and tokenized with spacy
    :param label_field: defines how to process the label associated with the text
    :param max_vocab_size: the maximum vocabulary size that will be built
    :param shuffle: if True, the dataset is shuffled before loading into the model
    :param random_state: if specified, seeds the random sampler when shuffling the data
    :param kwargs: any additional keyword arguments, currently unused
    """
    tmp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_filename = tmp_file.name
    # write df to a temp file
    data_df.to_csv(tmp_filename, sep=',', index=None)
    # load it as a CSVTextDataset
    csv_text_dataset = CSVTextDataset(os.path.dirname(tmp_filename), os.path.basename(tmp_filename),
                                      true_label=true_label, text_field=text_field, label_field=label_field,
                                      shuffle=shuffle, random_state=random_state, **kwargs)
    # delete tempfile
    os.remove(tmp_filename)
    return csv_text_dataset
