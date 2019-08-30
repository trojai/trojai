import logging

import cv2
import numpy as np
from numpy.random import RandomState

from .image_entity import ImageEntity, GenericImageEntity
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Module contains various classes that relate to size transformations of input objects
"""


class Resize(Transform):
    """
    Resizes an Entity
    """
    def __init__(self, new_size: tuple = (200, 200), interpolation: int = cv2.INTER_CUBIC) -> None:
        """
        Initialize the resizer object
        :param new_size: a tuple of the size in pixes for x and y dimensions
        :param interpolation: the interpolation method to resize the input Entity
        """
        self.new_size = new_size
        self.interpolation = interpolation

    def do(self, img_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the resizing
        :param img_obj: The input object to be resized according the specified configuration
        :param random_state_obj: ignored
        :return: The resized object
        """
        img_out = cv2.resize(img_obj.get_data(), self.new_size, interpolation=self.interpolation)
        mask_out = cv2.resize(img_obj.get_mask().astype(np.float32), self.new_size,
                              interpolation=self.interpolation).astype(bool)
        logger.info("Resized image of shape=%s to shape=%s using %d interpolation" %
                    (str(img_obj.get_data().shape), str(self.new_size), self.interpolation))
        return GenericImageEntity(img_out, mask_out)
