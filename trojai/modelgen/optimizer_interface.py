from abc import ABC, abstractmethod
from typing import Sequence
import logging

import torch.nn

from torch.utils.data import Dataset
from .training_statistics import EpochStatistics

logger = logging.getLogger(__name__)


class OptimizerInterface(ABC):
    """Object that performs training and testing of TrojAI models."""
    @abstractmethod
    def train(self, model: torch.nn.Module, data: Dataset, progress_bar_disable: bool,
              torch_dataloader_kwargs: dict = None) -> (torch.nn.Module, Sequence[EpochStatistics], int):
        """
        Train the given model using parameters in self.training_params
        :param model: (torch.nn.Module) The untrained Pytorch model
        :param data: (CSVDataset) Object containing training data, output 0 from TrojaiDataManager.load_data()
        :param progress_bar_disable: (bool) Don't display the progress bar if True
        :param torch_dataloader_kwargs: additional arguments to pass to PyTorch's DataLoader class
        :return: (torch.nn.Module, EpochStatistics) trained model, a sequence of EpochStatistics objects (one for
            each epoch), and the # of epochs with which the model was trained (useful for early stopping).
        """
        pass

    @abstractmethod
    def test(self, model: torch.nn.Module, clean_test_data: Dataset, triggered_test_data: Dataset,
             clean_test_triggered_labels_data: Dataset, torch_dataloader_kwargs) -> dict:
        """
        Perform whatever tests desired on the model with clean data and triggered data, return a dictionary of results.
        :param model: (torch.nn.Module) Trained Pytorch model
        :param clean_test_data: (CSVDataset) Object containing clean test data
        :param triggered_test_data: (CSVDataset or None) Object containing triggered test data, None if triggered data
            was not provided for testing
        :param clean_test_triggered_labels_data: triggered part of the training dataset but with correct labels; see
            DataManger.load_data for more information.
        :param torch_dataloader_kwargs: additional arguments to pass to PyTorch's DataLoader class
        :return: (dict) Dictionary of test accuracy results.
            Required key, value pairs are:
                clean_accuracy: (float in [0, 1]) classification accuracy on clean data
                clean_n_total: (int) number of examples in clean test set
            The following keys are optional, but should be used if triggered test data was provided
                triggered_accuracy: (float in [0, 1]) classification accuracy on triggered data
                triggered_n_total: (int) number of examples in triggered test set

        NOTE: This list may be augmented in the future to allow for additional test data collection.
        """
        pass

    @abstractmethod
    def get_device_type(self) -> str:
        """
        Return a string representation of the type of device used by the optimizer to train the model.
        """
        pass

    @abstractmethod
    def get_cfg_as_dict(self) -> dict:
        """
        Return a dictionary with key/value pairs that describe the parameters used to train the model.
        """
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        """
        Required for training on clusters. Return a deep copy of the optimizer.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """
        Required for training on clusters. Define how to chech if two optimizers are equal.
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def save(self, fname: str) -> None:
        """
        Save the optimizer to a file
        :param fname - the filename to save the optimizer to
        """
        pass

    @staticmethod
    @abstractmethod
    def load(fname: str):
        """
        Load an optimizer from disk and return it
        :param fname: the filename where the optimizer is serialized
        :return: The loaded optimizer
        """
        pass
