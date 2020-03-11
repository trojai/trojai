import copy
import logging
import os
from typing import Callable, Any, Union, Sequence
import types

import pandas as pd
from torch.utils.data import Dataset

from .constants import VALID_DATA_TYPES
from .datasets import CSVDataset, CSVTextDataset, csv_dataset_from_df, csv_textdataset_from_df
from .data_descriptions import DataDescription
from .data_configuration import DataConfiguration

logger = logging.getLogger(__name__)


class DataManager:
    """ Manages data from an experiment from trojai.datagen. """

    def __init__(self, experiment_path: str, train_file: Union[str, Sequence[str]], clean_test_file: str,
                 triggered_test_file: str = None,
                 data_type: str = 'image',
                 train_data_transform: Callable[[Any], Any] = lambda x: x,
                 train_label_transform: Callable[[int], int] = lambda y: y,
                 test_data_transform: Callable[[Any], Any] = lambda x: x,
                 test_label_transform: Callable[[int], int] = lambda y: y,
                 file_loader: Union[Callable[[str], Any], str] = 'default_image_loader',
                 shuffle_train=True, shuffle_clean_test=False, shuffle_triggered_test=False,
                 data_configuration: DataConfiguration = None,
                 custom_datasets: dict = None,
                 train_dataloader_kwargs: dict = None,
                 test_dataloader_kwargs: dict = None):
        """
        Initializes the DataManager object
        :param experiment_path: (str) absolute path to experiment data.
        :param train_file: (str) csv file name(s) of the training data. If iterable is provided, all will be trained
            on before model will be tested
        :param clean_test_file: (str) csv file name of the clean test data.
        :param triggered_test_file: (str) csv file name of the triggered test data.
        :param data_type: (str) can be 'image', 'text', or 'custom'.  The TrojaiDataManager uses this to determine how
                          to load the actual data and prepare it to be fed into the optimizer.
        :param train_data_transform: (function: any -> any) how to transform the training data (e.g. an image) to fit
            into the desired model and objective function; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param train_label_transform: (function: int->int) how to transform the label to the training data; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param test_data_transform: (function: any -> any) same as train_data_transform, but applied to validation and
            test data instead
            NOTE: Currently - this argument is only used if data_type='image'
        :param test_label_transform: (function: int->int) same as train_label_transform, but applied to validation and
            test data instead
            NOTE: Currently - this argument is only used if data_type='image'
        :param file_loader: (function: str->any or str) how to create the data object to pass into an architecture
            from a file path, or default loader to use. Options include: 'default_image_loader'
            default: 'default_image_loader'
            NOTE: Currently - this argument is only used if data_type='image'
        :param shuffle_train: (bool) shuffle the training data before training; default=True
        :param shuffle_clean_test: (bool) shuffle the clean test data; default=False
        :param shuffle_triggered_test (bool) shuffle the triggered test data; default=False
        :param data_configuration - a DataConfiguration object that might be useful for setting up
                how data is loaded
        :param custom_datasets - if data_type is 'custom', then the custom_datasets is a user implementation of
                torch.utils.data.Dataset.  We expect a dictionary of datasets, where the expected dictionary will
                look as follows:
                    {
                        'train': Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]],
                        'clean_test': torch.utils.data.Dataset,
                        'triggered_test': Union[torch.utils.data.Dataset, None]
                    }
        :param train_dataloader_kwargs: (dict) Keyword arguments to pass to the torch DataLoader object during training.
            See https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html for more documentation. If
            None, defaults will be used. Defaults depend on the optimizer used, but are likely something like:
                {batch_size: <batch size given in training config>, shuffle: True, pin_memory=<decided by optimizer>,
                 drop_last=True}
            NOTE: Setting values in this dictionary that are normally set by the optimizer will override them during
                training. Use with caution. We recommend only using the following keys: 'shuffle', 'num_workers',
                'pin_memory', and 'drop_last'.
        :param test_dataloader_kwargs: (dict) Similar to train_dataloader_kwargs, but for testing. Also, the default
            values for batch_size and shuffle will likely be 1 and False, respectively.
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
        self.data_loader = file_loader
        self.train_data_transform = train_data_transform
        self.train_label_transform = train_label_transform
        self.test_data_transform = test_data_transform
        self.test_label_transform = test_label_transform

        self.shuffle_train = shuffle_train
        self.shuffle_clean_test = shuffle_clean_test
        self.shuffle_triggered_test = shuffle_triggered_test

        self.data_configuration = data_configuration
        self.datasets = custom_datasets
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.test_dataloader_kwargs = test_dataloader_kwargs

        self.validate()

    def __deepcopy__(self, memodict={}):
        return DataManager(self.experiment_path, self.train_file, self.clean_test_file,
                           self.triggered_test_file, self.data_type, copy.deepcopy(self.train_data_transform),
                           copy.deepcopy(self.train_label_transform), copy.deepcopy(self.test_data_transform),
                           copy.deepcopy(self.test_label_transform), copy.deepcopy(self.data_loader),
                           self.shuffle_train, self.shuffle_clean_test, self.shuffle_triggered_test,
                           self.data_configuration, self.datasets, self.train_dataloader_kwargs,
                           self.test_dataloader_kwargs)

    def __eq__(self, other):
        if self.experiment_path == other.experiment_path and self.train_file == other.train_file and \
                self.clean_test_file == other.clean_test_file and \
                self.triggered_test_file == other.triggered_test_file and \
                self.data_type == other.data_type and \
                self.train_data_transform == other.train_data_transform and \
                self.train_label_transform == other.train_label_transform and \
                self.test_data_transform == other.test_data_transform and \
                self.test_label_transform == other.test_label_transform and \
                self.data_loader == other.data_loader and self.shuffle_train == other.shuffle_train and \
                self.shuffle_clean_test == other.shuffle_clean_test and \
                self.shuffle_triggered_test == other.shuffle_triggered_test and \
                self.data_configuration == other.data_configuration and \
                self.train_dataloader_kwargs == other.train_dataloader_kwargs and \
                self.test_dataloader_kwargs == other.test_dataloader_kwargs:
            # Note: when we compare callables, we simply compare whether the callable is the same reference in memory
            #  or not.  This means that if two callables are functionally equivalent, but are different object
            #  references then the equality comparison will fail
            #  This note also pertains to comparing the dataset_obj, which is a torch.utils.data.Dataset object!

            # TODO: compare datasets

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
            first_dataset = CSVDataset(self.experiment_path, self.train_file[0],
                                       data_transform=self.train_data_transform,
                                       label_transform=self.train_label_transform,
                                       data_loader=self.data_loader,
                                       shuffle=self.shuffle_train)
            train_dataset = (first_dataset if ii == 0 else CSVDataset(self.experiment_path, self.train_file[ii],
                                                                      data_transform=self.train_data_transform,
                                                                      label_transform=self.train_label_transform,
                                                                      data_loader=self.data_loader,
                                                                      shuffle=self.shuffle_train)
                             for ii in range(len(self.train_file)))

            if self.clean_test_file:
                clean_test_dataset = CSVDataset(self.experiment_path, self.clean_test_file,
                                                data_transform=self.test_data_transform,
                                                label_transform=self.test_label_transform,
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
            if self.triggered_test_file:
                triggered_test_dataset = CSVDataset(self.experiment_path, self.triggered_test_file,
                                                    data_transform=self.test_data_transform,
                                                    label_transform=self.test_label_transform,
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

            # This dataset contains the subset of clean examples which also have a triggered counterpart.
            # For example, if an MNIST dataset was created with triggered examples only for labels 4 and 5,
            # then this dataset is the subset of data with labels 4 and 5 that don't have the triggers.
            if clean_test_dataset and triggered_test_dataset:
                triggered_classes = triggered_test_dataset.data_df['true_label'].unique()

                clean_test_df_triggered_classes_only = clean_test_dataset.data_df[
                    clean_test_dataset.data_df['true_label'].isin(triggered_classes)]
                clean_test_triggered_classes_dataset = csv_dataset_from_df(self.experiment_path,
                                                                           clean_test_df_triggered_classes_only,
                                                                           data_transform=self.test_data_transform,
                                                                           label_transform=self.test_label_transform,
                                                                           data_loader=self.data_loader,
                                                                           shuffle=self.shuffle_triggered_test)
            else:
                clean_test_triggered_classes_dataset = None

            # nothing to fill in at the moment for image, we can update as needed
            if isinstance(train_dataset, types.GeneratorType):
                train_dataset_desc = first_dataset.get_data_description()
            else:
                train_dataset_desc = train_dataset.get_data_description()
            if clean_test_dataset:
                clean_test_dataset_desc = clean_test_dataset.get_data_description()
            else:
                clean_test_dataset_desc = None
            if triggered_test_dataset:
                triggered_test_dataset_desc = triggered_test_dataset.get_data_description()
            else:
                triggered_test_dataset_desc = None

        elif self.data_type == 'text':
            if len(self.train_file) > 1:
                msg = "Sequential Training not supported for Text datatype!"
                logger.error(msg)
                raise ValueError(msg)
            # ensure a DataDescription is set for text data!
            if self.data_configuration is None:
                msg = "data_configuration object needs to be set for Text data processing!"
                logger.error(msg)
                raise ValueError(msg)

            logger.info("Loading Training Dataset")
            train_dataset = CSVTextDataset(self.experiment_path, self.train_file[0], shuffle=self.shuffle_train,
                                           text_field_kwargs=self.data_configuration.text_field_kwargs,
                                           label_field_kwargs=self.data_configuration.label_field_kwargs
                                           )
            train_dataset.build_vocab(self.data_configuration.embedding_vectors_cfg,
                                      self.data_configuration.max_vocab_size,
                                      self.data_configuration.text_field_kwargs['use_vocab'])

            # pass in the learned vocabulary from the training data to the clean test dataset

            if self.clean_test_file:
                clean_test_dataset = CSVTextDataset(self.experiment_path, self.clean_test_file,
                                                    text_field=train_dataset.text_field,
                                                    label_field=train_dataset.label_field,
                                                    text_field_kwargs=self.data_configuration.text_field_kwargs,
                                                    label_field_kwargs=self.data_configuration.label_field_kwargs,
                                                    shuffle=self.shuffle_clean_test)
                if len(clean_test_dataset) == 0:
                    msg = 'Clean Test Dataset was empty and will be skipped...'
                    logger.info(msg)
            else:
                clean_test_dataset = None
                msg = 'Clean Test Dataset was empty and will be skipped...'
                logger.info(msg)

            if self.triggered_test_file:
                logger.info("Loading Triggered Test Dataset")
                # pass in the learned vocabulary from the training data to the triggered test dataset
                triggered_test_dataset = CSVTextDataset(self.experiment_path, self.triggered_test_file,
                                                        text_field=train_dataset.text_field,
                                                        label_field=train_dataset.label_field,
                                                        text_field_kwargs=self.data_configuration.text_field_kwargs,
                                                        label_field_kwargs=self.data_configuration.label_field_kwargs,
                                                        shuffle=self.shuffle_triggered_test)
                if len(triggered_test_dataset) == 0:
                    msg = 'Triggered Dataset was empty, testing on triggered data will be skipped...'
                    logger.info(msg)
                    triggered_test_dataset = None
            else:
                triggered_test_dataset = None

            # figure out which classes are triggered, and subset the clean dataset for just those classes,
            # as another metric of interest
            if clean_test_dataset and triggered_test_dataset:
                triggered_classes = triggered_test_dataset.data_df['true_label'].unique()
                clean_test_df_triggered_classes_only = clean_test_dataset.data_df[
                    clean_test_dataset.data_df['true_label'].isin(triggered_classes)]
                clean_test_triggered_classes_dataset = csv_textdataset_from_df(clean_test_df_triggered_classes_only,
                                                                               text_field=train_dataset.text_field,
                                                                               label_field=train_dataset.label_field,
                                                                               text_field_kwargs=self.data_configuration.text_field_kwargs,
                                                                               label_field_kwargs=self.data_configuration.label_field_kwargs,
                                                                               shuffle=self.shuffle_triggered_test)
            else:
                clean_test_triggered_classes_dataset = None

            train_dataset_desc = train_dataset.get_data_description()
            if clean_test_dataset and len(clean_test_dataset) > 0:
                clean_test_dataset_desc = clean_test_dataset.get_data_description()
            else:
                clean_test_dataset_desc = None
            if triggered_test_dataset and len(triggered_test_dataset) > 0:
                triggered_test_dataset_desc = triggered_test_dataset.get_data_description()
            else:
                triggered_test_dataset_desc = None

        elif self.data_type == 'custom':
            train_dataset = self.datasets['train']
            clean_test_dataset = self.datasets['clean_test']
            triggered_test_dataset = self.datasets.get('triggered_test')
            clean_test_triggered_classes_dataset = self.datasets.get('clean_test_triggered_classes_dataset')
            if train_dataset:
                train_dataset_desc = train_dataset.get_data_description()
            else:
                train_dataset_desc = None
            if clean_test_dataset:
                clean_test_dataset_desc = clean_test_dataset.get_data_description()
            else:
                clean_test_dataset_desc = None
            if triggered_test_dataset:
                triggered_test_dataset_desc = triggered_test_dataset.get_data_description()
            else:
                triggered_test_dataset_desc = None

        else:
            msg = "Unsupported data_type argument provided"
            logger.error(msg)
            raise NotImplementedError(msg)

        if clean_test_triggered_classes_dataset:
            clean_test_triggered_classes_dataset_desc = clean_test_triggered_classes_dataset.get_data_description()
        else:
            clean_test_triggered_classes_dataset_desc = None

        return train_dataset, clean_test_dataset, triggered_test_dataset, clean_test_triggered_classes_dataset, \
            train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc, clean_test_triggered_classes_dataset_desc

    def validate(self) -> None:
        """
        Validate the construction of the TrojaiDataManager object
        :return: None

        TODO:
         [ ] - think about whether the contents of the files passed into the DataManager should be validated,
               in addition to simply checking for existence, which is what is done now
        """
        if self.train_dataloader_kwargs is not None and not isinstance(self.train_dataloader_kwargs, dict):
            msg = "train_dataloader_kwargs must be a dictionary!"
            logger.error(msg)
            raise ValueError(msg)
        if self.test_dataloader_kwargs is not None and not isinstance(self.test_dataloader_kwargs, dict):
            msg = "train_dataloader_kwargs must be a dictionary!"
            logger.error(msg)
            raise ValueError(msg)

        if self.data_type == 'custom':
            if self.datasets is None:
                msg = "dataset_obj must not be None if data_type is set to Custom"
                logger.error(msg)
                raise ValueError(msg)
            elif isinstance(self.datasets, dict):
                required_keys_to_test = ['train', 'clean_test']
                optional_keys_to_test = ['triggered_test']
                dd_keys_to_test = ['train_data_description', 'clean_test_data_description',
                                   'triggered_test_data_description']
                for k in required_keys_to_test:
                    if k in self.datasets:
                        if isinstance(self.datasets[k], Dataset):
                            pass
                        else:
                            msg = "The expected type of value for key:" + k + ' is Dataset'
                            logger.error(msg)
                            raise ValueError(msg)
                    else:
                        msg = 'Expected key:' + k + ' in datasets dictionary'
                        logger.error(msg)
                        raise ValueError(msg)
                for k in optional_keys_to_test:
                    if k in self.datasets:
                        if isinstance(self.datasets[k], Dataset):
                            pass
                        else:
                            msg = "The expected type of value for key:" + k + ' is Dataset'
                            logger.error(msg)
                            raise ValueError(msg)
                for k in dd_keys_to_test:
                    if k in self.datasets:
                        if self.datasets[k] is None or isinstance(self.datasets[k], DataDescription):
                            pass
                        else:
                            msg = "Expected type for key:" + k + " is either None or of type DataDescription"
                            logger.error(msg)
                            raise ValueError(msg)
            else:
                msg = "dataset_obj must be of type dict with minium keys of train and clean_test"
                logger.error(msg)
                raise ValueError(msg)
        else:
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
            if not callable(self.train_data_transform):
                raise TypeError("Expected a function for argument 'train_data_transform', "
                                "instead got type: {}".format(type(self.train_data_transform)))
            if not callable(self.train_label_transform):
                raise TypeError("Expected a function for argument 'train_label_transform', "
                                "instead got type: {}".format(type(self.train_label_transform)))
            if not callable(self.test_data_transform):
                raise TypeError("Expected a function for argument 'test_data_transform', "
                                "instead got type: {}".format(type(self.test_data_transform)))
            if not callable(self.test_label_transform):
                raise TypeError("Expected a function for argument 'test_label_transform', "
                                "instead got type: {}".format(type(self.test_label_transform)))
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
