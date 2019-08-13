from typing import Callable, Union
import os
import logging
import cv2
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchtext.data

logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    def __init__(self, path_to_data, csv_filename, true_label=False, path_to_csv=None, shuffle=False, random_state=None,
                 data_loader: Union[str, Callable] = 'default_image_loader', data_transform=lambda x: x, label_transform=lambda l: l):
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
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path_to_data, csv_filename, text_field=None, label_field=None, max_vocab_size=25000,
                 shuffle=False, random_state=None, **kwargs):
        """
        Initialize the CSV Text Dataset object
        :param path_to_data:
        :param csv_filename:
        :param text_field:
        :param label_field:
        :param max_vocab_size:
        :param shuffle:
        :param random_state:
        :param kwargs:

        TODO:
         [ ] - parallelize reading in data from disk
         [ ] - revisit reading entire corpus into memory
        """
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
            label = row['train_label']
            with open(os.path.join(path_to_data, fname), 'r') as f:
                z = f.readlines()
                text = ' '.join(z)
            examples.append(torchtext.data.Example.fromlist([text, label], fields))

        super(CSVTextDataset, self).__init__(examples, fields, **kwargs)
