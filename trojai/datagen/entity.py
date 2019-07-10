import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

"""
Defines a generic Entity object, and an Entity convenience wrapper for creating Entities from numpy arrays.  
"""

DEFAULT_DTYPE = np.uint8


class Entity(ABC):
    """
    An Entity is a generalization of a synthetic object.  It could stand alone, or a composition of multiple entities.
    An Entity is composed of some data (represented by a numpy.ndarray), and an associated "valid" mask (numpy.ndarray).
    While the data can be a multi-channel, the mask must be a matrix of shape image.shape[0:2].  In other words,
    the mask applies to each channel of the data in the same way.  This is currently done for ease, but could be
    expanded such that a mask is specified for each channel of the image, if necessary.
    See the README for further details on how Entity objects are intended to be used in the TrojAI pipeline.
    """
    @abstractmethod
    def get_data(self) -> np.ndarray:
        """
        Get the data associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        pass

    @abstractmethod
    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        pass


class GenericEntity(Entity):
    """
    A class which allows one to easily instantiate a generic Entity object with an image and associated mask
    """
    def __init__(self, data: np.ndarray, mask: np.ndarray = None) -> None:
        """
        Initialize the GenericEntity object, given an input image and associated mask
        :param data: The input image to be wrapped into an Entity
        :param mask: The associated mask to be wrapped into an Entity
        """
        self.pattern = data
        if mask is None:
            self.mask = np.ones(data.shape[0:2]).astype(bool)
        elif isinstance(mask, np.ndarray):
            if mask.shape[0:2] == self.pattern.shape[0:2]:
                self.mask = mask.astype(bool)
            else:
                msg = "Unknown Mask input - must be either None of a numpy.ndarray of same shape as arr_input"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "Unknown Mask input - must be either None of a numpy.ndarray of same shape as arr_input"
            logger.error(msg)
            raise ValueError(msg)

    def get_data(self) -> np.ndarray:
        """
        Get the data associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
