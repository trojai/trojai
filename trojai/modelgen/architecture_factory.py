import logging
from abc import ABC, abstractmethod

import torch.nn

logger = logging.getLogger(__name__)


class ArchitectureFactory(ABC):
    """ Factory object that returns architectures (untrained models) for training. """
    @abstractmethod
    def new_architecture(self, **kwargs) -> torch.nn.Module:
        """
        Returns a new architecture (untrained model)
        :return: an untrained torch.nn.Module
        """
        pass

    def __eq__(self, other):
        """
        Compares two Architecture factories by comparing the string representations of the Architectures
        returned by the new_architecture() function
        :param other: the ArchitectureFactory to compare against
        :return: boolean indicating whether the architectures are the same or not
        """

        my_arch_instance = self.new_architecture()
        other_arch_instance = other.new_architecture()
        # only keep the unique elements that are not part of the nn.Module
        dir_nn_module = set(dir(torch.nn.Module))
        dir_my_arch = set(dir(my_arch_instance)) - dir_nn_module
        dir_other_arch = set(dir(other_arch_instance)) - dir_nn_module

        if len(dir_my_arch) == len(dir_other_arch):
            for item in dir_my_arch:
                if item in dir_other_arch:
                    if item[0] != '_':
                        # compare the actual objects
                        my_item = getattr(my_arch_instance, item)
                        other_item = getattr(other_arch_instance, item)
                        # NOTE: here, we check whether the arch-factory is the same based on the string representation
                        #  of a returned architecture.
                        #  this could easily be error-prone, need to revisit how to make this more robust
                        if str(my_item) != str(other_item):
                            return False
                else:
                    return False
        else:
            return False

        return True
