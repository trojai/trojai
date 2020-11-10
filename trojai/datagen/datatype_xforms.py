import logging

import numpy as np
from numpy.random import RandomState

from .image_entity import GenericImageEntity, ImageEntity
from .transform_interface import ImageTransform

logger = logging.getLogger(__name__)

"""
Defines data type transformations that may need to occur when processing different data sources
"""


class ToTensorXForm(ImageTransform):
    """
    Transformation which defines the conversion of an input array to a tensor of a specified # of dimensions
    """
    def __init__(self, num_dims: int = 3) -> None:
        """
        Create the transformer object
        :param num_dims: the number of dimensions to convert the input into
        """
        self.num_dims = num_dims

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual to->tensor conversion
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        old_shape = img.shape
        num_img_dims = len(img.shape)
        if num_img_dims >= self.num_dims:
            return input_obj
        else:
            num_dims_to_add = self.num_dims-num_img_dims
            for ii in range(num_dims_to_add):
                img = np.expand_dims(img, axis=len(img.shape))
        new_shape = img.shape
        logger.debug("Converted input entity from shape=%s to %s" % (str(old_shape), str(new_shape)))
        # make a new Entity object and return
        return GenericImageEntity(img, input_obj.get_mask())
