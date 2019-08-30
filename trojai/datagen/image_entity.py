import logging
from abc import abstractmethod

import numpy as np

from .entity import Entity

logger = logging.getLogger(__name__)

"""
Defines a generic Entity object, and an Entity convenience wrapper for creating Entities from numpy arrays.  
"""

DEFAULT_DTYPE = np.uint8


class ImageEntity(Entity):
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        pass


class GenericImageEntity(ImageEntity):
    """
    A class which allows one to easily instantiate an ImageEntity object with an image and associated mask
    """
    def __init__(self, data: np.ndarray, mask: np.ndarray = None) -> None:
        """
        Initialize the GenericImageEntity object, given an input image and associated mask
        :param data: The input image to be wrapped into an ImageEntity
        :param mask: The associated mask to be wrapped into an ImageEntity
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
        Get the data associated with the ImageEntity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the ImageEntity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
