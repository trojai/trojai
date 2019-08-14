from abc import ABC, abstractmethod
from typing import Sequence
import logging

import torch.nn

from .data_manager import CSVDataset
from .training_statistics import EpochStatistics

logger = logging.getLogger(__name__)


class OptimizerInterface(ABC):
    @abstractmethod
    def train(self, model: torch.nn.Module, data: CSVDataset, train_val_split: float) \
            -> (torch.nn.Module, Sequence[EpochStatistics]):
        """
        Train the given model using parameters in self.training_params
        :param model: (torch.nn.Module) The untrained Pytorch model
        :param data: (CSVDataset) Object containing training data, output 0 from TrojaiDataManager.load_data()
        :param train_val_split: (float) percentage of data that should be used for validation
        :return: (torch.nn.Module) trained model
        """
        pass

    """ Object that performs training and testing of TrojAI models. """

    @abstractmethod
    def test(self, model, clean_test_data, triggered_test_data) -> dict:
        """
        Perform whatever tests desired on the model with clean data and triggered data. Still unknown what this should
        do... Default Optimizer currently prints results and returns them in a dictionary.
        :param model: (torch.nn.Module) Trained Pytorch model
        :param clean_test_data: (CSVDataset) Object containing clean test data, output 1 from
            TrojaiDataManager.load_data()
        :param triggered_test_data: (CSVDataset or None) Object containing triggered test data, output 2 from
            TrojaiDataManager.load_data(), None if triggered data was not provided
        :return: (?) test results
        """
        pass

    @abstractmethod
    def get_device_type(self) -> str:
        """
        Returns a string representation of the device used by the optimizer to train the model
        :return: a string representation of the device used by the optimizer to train the model
        """
        pass

    @abstractmethod
    def get_cfg_as_dict(self) -> dict:
        """
        Returns a dictionary with key/value pairs that describe the parameters used to train the model.
        :return:
        """
        pass

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def save(self, fname: str) -> None:
        """
        Save the optimizer to a file
        :param fname - the filename to save the optimizer to
        :return: None
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
