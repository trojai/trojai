import copy
import logging
import os
from typing import Callable, Any, Union, Sequence

import pandas as pd
import torch

from .constants import VALID_DATA_TYPES
from .datasets import CSVDataset, CSVTextDataset

logger = logging.getLogger(__name__)


class DataManager:
    """ Manages data from an experiment from trogai.datagen. """
    def __init__(self, experiment_path: str, train_file: Union[str, Sequence[str]], clean_test_file: str,
                 triggered_test_file: str = None,
                 data_type: str = 'image',
                 data_transform: Callable[[Any], Any] = (lambda x: x),
                 label_transform: Callable[[int], int] = lambda y: y,
                 data_loader: Union[Callable[[str], Any], str] = 'default_image_loader',
                 shuffle_train=True, shuffle_clean_test=False, shuffle_triggered_test=False):
        """
        Initializes the DataManager object
        :param experiment_path: (str) absolute path to experiment data.
        :param train_file: (str) csv file name(s) of the training data. If iterable is provided, all will be trained
            on before model will be tested
        :param clean_test_file: (str) csv file name of the clean test data.
        :param triggered_test_file: (str) csv file name of the triggered test data.
        :param data_type: (str) can be 'image' or 'text'.  The TrojaiDataManager uses this to determine how to load
                          the actual data at a more fundamental level than the data_loader argument.
        :param data_transform: (function: any -> any) how to transform the data (e.g. and image) to fit into the
            desired model and objective function; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param label_transform: (function: int->int) how to transform the label to the data; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param data_loader: (function: str->any or str) how to create the data object to pass into an architecture
            from a file path, or default loader to use. Options include: 'default_image_loader'
            default: 'default_image_loader'
            NOTE: Currently - this argument is only used if data_type='image'
        :param shuffle_train: (bool) shuffle the training data before training; default=True
        :param shuffle_clean_test: (bool) shuffle the clean test data; default=False
        :param shuffle_triggered_test (bool) shuffle the triggered test data; default=False
        """

        self.experiment_path = experiment_path
        try:
            iter(train_file)
        except TypeError:
            pass
        if type(train_file) == str:
            train_file = [train_file]
        self.train_file = train_file
        self.clean_test_file = clean_test_file
        self.triggered_test_file = triggered_test_file

        self.data_type = data_type
        self.data_loader = data_loader
        self.data_transform = data_transform
        self.label_transform = label_transform

        self.shuffle_train = shuffle_train
        self.shuffle_clean_test = shuffle_clean_test
        self.shuffle_triggered_test = shuffle_triggered_test

        self.validate()

    def __deepcopy__(self, memodict={}):
        return DataManager(self.experiment_path, self.train_file, self.clean_test_file,
                           self.triggered_test_file, self.data_type, copy.deepcopy(self.data_transform),
                           copy.deepcopy(self.label_transform), copy.deepcopy(self.data_loader),
                           self.shuffle_train, self.shuffle_clean_test, self.shuffle_triggered_test)

    def __eq__(self, other):
        if self.experiment_path == other.experiment_path and self.train_file == other.train_file and \
           self.clean_test_file == other.clean_test_file and self.triggered_test_file == other.triggered_test_file and \
           self.data_type == other.data_type and \
           self.data_transform == other.data_transform and self.label_transform == other.label_transform and \
           self.data_loader == other.data_loader and self.shuffle_train == other.shuffle_train and \
           self.shuffle_clean_test == other.shuffle_clean_test and \
           self.shuffle_triggered_test == other.shuffle_triggered_test:
            # Note: when we compare callables, we simply compare whether the callable is the same reference in memory
            #  or not.  This means that if two callables are functionally equivalent, but are different object
            #  references then the equality comparison will fail
            return True
        else:
            return False

    def load_data(self):
        """
        Load experiment data as given from initialization.
        :return: Objects containing training and test, and triggered data if it was provided.

        TODO:
         [ ] - extend the text data-type to have more input arguments, for example the tokenizer and FIELD options
         [ ] - need to support sequential training for text datasets
        """
        if self.data_type == 'image':
            logger.info("Loading Training Dataset")
            train_dataset = (CSVDataset(self.experiment_path, f,
                                        data_transform=self.data_transform,
                                        label_transform=self.label_transform,
                                        data_loader=self.data_loader,
                                        shuffle=self.shuffle_train) for f in self.train_file)
            if self.clean_test_file is not None:
                clean_test_dataset = CSVDataset(self.experiment_path, self.clean_test_file,
                                                data_transform=self.data_transform,
                                                label_transform=self.label_transform,
                                                data_loader=self.data_loader,
                                                shuffle=self.shuffle_clean_test)
                if len(clean_test_dataset) == 0:
                    clean_test_dataset = None
                    msg = 'Clean Test Dataset was empty and will be skipped...'
                    logger.info(msg)
            else:
                clean_test_dataset = None
                msg = 'Clean Test Dataset was empty and will be skipped...'
                logger.info(msg)
            if self.triggered_test_file is not None:
                triggered_test_dataset = CSVDataset(self.experiment_path, self.triggered_test_file,
                                                    data_transform=self.data_transform,
                                                    label_transform=self.label_transform,
                                                    data_loader=self.data_loader,
                                                    shuffle=self.shuffle_triggered_test)
                if len(triggered_test_dataset) == 0:
                    triggered_test_dataset = None
                    msg = 'Triggered Dataset was empty, testing on triggered data will be skipped...'
                    logger.info(msg)
            else:
                triggered_test_dataset = None
                msg = 'Triggered Dataset was empty, testing on triggered data will be skipped...'
                logger.info(msg)

        elif self.data_type == 'text':
            if len(self.train_file) > 1:
                msg = "Sequential Training not supported for Text datatype!"
                logger.error(msg)
                raise ValueError(msg)
            logger.info("Loading Training Dataset")
            train_dataset = CSVTextDataset(self.experiment_path, self.train_file[0], shuffle=self.shuffle_train)
            embedding_vectors_cfg = "glove.6B.100d"
            logger.info("Building Vocabulary from training data using: " + str(embedding_vectors_cfg) +
                        " with a max vocab size=" + str(train_dataset.max_vocab_size) + " !")
            train_dataset.text_field.build_vocab(train_dataset,
                                                 max_size=train_dataset.max_vocab_size,
                                                 vectors="glove.6B.100d",
                                                 unk_init=torch.Tensor.normal_)
            train_dataset.label_field.build_vocab(train_dataset)
            logger.info("Loading Clean Test Dataset")
            # pass in the learned vocabulary from the training data to the clean test dataset

            if self.clean_test_file is not None:
                clean_test_dataset = CSVTextDataset(self.experiment_path, self.clean_test_file,
                                                    text_field=train_dataset.text_field,
                                                    label_field=train_dataset.label_field,
                                                    shuffle=self.shuffle_clean_test)
                if len(clean_test_dataset) == 0:
                    msg = 'Clean Test Dataset was empty and will be skipped...'
                    logger.info(msg)
            else:
                msg = 'Clean Test Dataset was empty and will be skipped...'
                logger.info(msg)
            if self.triggered_test_file is not None:
                logger.info("Loading Triggered Test Dataset")
                # pass in the learned vocabulary from the training data to the triggered test dataset
                triggered_test_dataset = CSVTextDataset(self.experiment_path, self.triggered_test_file,
                                                        text_field=train_dataset.text_field,
                                                        label_field=train_dataset.label_field,
                                                        shuffle=self.shuffle_triggered_test)
                if len(triggered_test_dataset) == 0:
                    msg = 'Triggered Dataset was empty, testing on triggered data will be skipped...'
                    logger.info(msg)
                    triggered_test_dataset = None
            else:
                triggered_test_dataset = None
        else:
            msg = "Unsupported data_type argument provided"
            logger.error(msg)
            raise NotImplementedError(msg)

        return train_dataset, clean_test_dataset, triggered_test_dataset

    def validate(self) -> None:
        """
        Validate the construction of the TrojaiDataManager object
        :return: None

        TODO:
         [ ] - think about whether the contents of the files passed into the DataManager should be validated,
               in addition to simply checking for existence, which is what is done now
        """
        # check types
        if type(self.experiment_path) != str:
            raise TypeError("Expected type 'string' for argument 'experiment_path', "
                            "instead got type: {}".format(type(self.experiment_path)))
        for fn in self.train_file:
            if type(fn) != str:
                raise TypeError("Expected string or Iterable[string] for argument 'train_file', "
                                "instead got type: {}".format(type(fn)))
        if type(self.clean_test_file) != str:
            raise TypeError("Expected type 'string' for argument 'clean_test_file', "
                            "instead got type: {}".format(type(self.clean_test_file)))
        if self.triggered_test_file is not None and type(self.triggered_test_file) != str:
            raise TypeError("Expected type 'string' for argument 'triggered_test_file', "
                            "instead got type: {}".format(type(self.triggered_test_file)))
        if not callable(self.data_transform):
            raise TypeError("Expected a function for argument 'data_transform', "
                            "instead got type: {}".format(type(self.data_transform)))
        if not callable(self.label_transform):
            raise TypeError("Expected a function for argument 'label_transform', "
                            "instead got type: {}".format(type(self.label_transform)))
        if not callable(self.data_loader) and type(self.data_loader) != str:
            raise TypeError("Expected a function or string for argument 'data_loader', "
                            "instead got type: {}".format(type(self.data_loader)))
        if not type(self.shuffle_train) == bool:
            raise TypeError("Expected type 'bool' for argument 'shuffle_train', "
                            "instead got type: {}".format(type(self.shuffle_train)))
        if not type(self.shuffle_clean_test) == bool:
            raise TypeError("Expected type 'bool' for argument 'shuffle_clean_test', "
                            "instead got type: {}".format(type(self.shuffle_clean_test)))
        if not type(self.shuffle_triggered_test) == bool:
            raise TypeError("Expected type 'bool' for argument 'shuffle_triggered_test', "
                            "instead got type: {}".format(type(self.shuffle_triggered_test)))

        # check if files and directories exist
        if not os.path.isdir(self.experiment_path):
            raise FileNotFoundError("{} directory was not found...".format(self.experiment_path))
        for f in self.train_file:
            if not os.path.isfile(os.path.join(self.experiment_path, f)):
                raise FileNotFoundError("{} training file was not found".format(f))
        if not os.path.isfile(os.path.join(self.experiment_path, self.clean_test_file)):
            raise FileNotFoundError("{} clean test file file was not found".format(self.clean_test_file))
        if self.triggered_test_file is not None and \
                not os.path.isfile(os.path.join(self.experiment_path, self.triggered_test_file)):
            raise FileNotFoundError("{} triggered test file file was not found".format(self.triggered_test_file))

        # check if training data is empty
        for f in self.train_file:
            train_path = os.path.join(self.experiment_path, f)
            train_df = pd.read_csv(train_path)
            if len(train_df) == 0:
                err_msg = "'train_file' {} is empty".format(train_path)
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        clean_test_path = os.path.join(self.experiment_path, self.clean_test_file)
        clean_test_df = pd.read_csv(clean_test_path)
        if len(clean_test_df) == 0:
            err_msg = "'clean_test_file' is empty"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if not isinstance(self.data_type, str):
            msg = "data_type argument must be one of the following: " + str(VALID_DATA_TYPES)
            logger.error(msg)
            raise ValueError(msg)
        else:
            if self.data_type not in VALID_DATA_TYPES:
                msg = "Unsupported data_type argument provided"
                logger.error(msg)
                raise ValueError(msg)
