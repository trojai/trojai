import logging

from numpy.random import RandomState
from trojai.datagen.image_conversion_utils import gray_to_rgb, rgba_to_rgb, rgb_to_rgba

from .image_entity import ImageEntity, GenericImageEntity
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Defines several transformations related to static (non-random) color manipulation
"""


class GrayscaleToRGBXForm(Transform):
    """ Converts an 3-channel grayscale image to RGB

    """
    def __init__(self) -> None:
        """
        Creates the object to perform the transformation.
        """
        pass

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Convert the input object from 3-channel grayscale to RGB
        :param input_obj: Entity to be colorized
        :param random_state_obj: ignored
        :return: The colorized entity
        """
        img = input_obj.get_data()
        rgb_img = gray_to_rgb(img)
        logger.debug("Converted input object from 3-channel grayscale to RGB")
        return GenericImageEntity(rgb_img, input_obj.get_mask())


class RGBAtoRGB(Transform):
    """ Converts input Entity from RGBA to RGB
    """
    def __init__(self) -> None:
        """
        Create the transformer object
        """
        pass

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the RGBA to RGB transformation
        :param input_obj: the Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        rgb_img, alpha_ch = rgba_to_rgb(img)
        logger.debug("Converted input object from RGBA to RGB")
        return GenericImageEntity(rgb_img, input_obj.get_mask())


class RGBtoRGBA(Transform):
    """ Converts input Entity from RGB to RGBA

    """
    def __init__(self) -> None:
        """
        Create the transformer object
        """
        pass

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the RGBA to RGB transformation
        :param input_obj: the Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        rgba_img = rgb_to_rgba(img)
        logger.debug("Converted input object from RGB to RGBA")
        return GenericImageEntity(rgba_img, input_obj.get_mask())
