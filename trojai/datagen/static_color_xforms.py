import logging

import cv2
from numpy.random import RandomState

from .entity import Entity, GenericEntity
from .transform import Transform

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

    def do(self, input_obj: Entity, random_state_obj: RandomState) -> Entity:
        """
        Convert the input object from 3-channel grasycale to RGB
        :param input_obj: Entity to be colorized
        :param random_state_obj: ignored
        :return: The colorized entity
        """
        img = input_obj.get_data()
        if len(img.shape) > 3 or len(img.shape) < 2:
            msg = "Input image doesn't seem to be grayscale!"
            logger.error(msg)
            raise ValueError(msg)

        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        logger.info("Converted input object from 3-channel grayscale to RGB")
        return GenericEntity(img_out, input_obj.get_mask())


class RGBAtoRGB(Transform):
    """ Converts input Entity from RGBA to RGB
    """
    def __init__(self) -> None:
        """
        Create the transformer object
        """
        pass

    def do(self, input_obj: Entity, random_state_obj: RandomState) -> Entity:
        """
        Perform the RGBA to RGB transformation
        :param input_obj: the Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        if len(img.shape) > 2:
            if img.shape[2] == 3:
                return input_obj
            elif img.shape[2] == 4:
                # convert the image from RGBA2RGB
                # same as BGRA2BGR, see: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
                output_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                logger.info("Converted input object from RGBA to RGB")
                return GenericEntity(output_img, input_obj.get_mask())
            else:
                msg = "Unknown Image Format!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "Input image doesn't have enough channels!"
            logger.error(msg)
            raise ValueError(msg)


class RGBtoRGBA(Transform):
    """ Converts input Entity from RGB to RGBA

    """
    def __init__(self) -> None:
        """
        Create the transformer object
        """
        pass

    def do(self, input_obj: Entity, random_state_obj: RandomState) -> Entity:
        """
        Perform the RGBA to RGB transformation
        :param input_obj: the Entity to be transformed
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        if len(img.shape) > 2:
            if img.shape[2] == 4:
                # already have alpha layer, pass through
                return input_obj
            elif img.shape[2] == 3:
                # convert the image from RGB2RGBA
                output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                logger.info("Converted input object from RGB to RGBA")
                return GenericEntity(output_img, input_obj.get_mask())
            else:
                msg = "Unknown Image Format!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "Input image # channels unsupported!"
            logger.error(msg)
            raise ValueError(msg)
