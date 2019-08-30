import logging
from typing import Sequence, Dict

import skimage.transform
import numpy as np
from numpy.random import RandomState

from .image_entity import GenericImageEntity, ImageEntity
from .transform_interface import ImageTransform

logger = logging.getLogger(__name__)

"""
Module defines several affine transforms using various libraries to perform  the actual transformation operation 
specified.
"""


class RotateXForm(ImageTransform):
    """Implements a rotation of an Entity by a specified angle amount.

    """
    def __init__(self, angle: int = 90, args: tuple = (), kwargs: dict = None) -> None:
        """
        Creates a Rotator Transform object
        :param angle: The degree amount to rotate (in degrees, not radians!)
        :param args: any additional arguments to pass to skimage.transform.rotate
        :param kwargs: any keyword arguments to pass to skimage.transform.rotate
        """
        self.rotation_angle = angle
        self.args = args
        if kwargs is None:
            self.kwargs = {'preserve_range': True}
        else:
            if 'preserve_range' in kwargs and not kwargs['preserve_range']:
                msg = "preserve_range cannot be set to False!"
                logger.error(msg)
                raise ValueError(msg)
            self.kwargs = kwargs

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Performs the rotation specified by the RotateXForm object on an input
        :param input_obj: The Entity to be rotated
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        mask = input_obj.get_mask()

        logger.info("Applying %0.02f rotation to image via skimage.transform.rotate" % (self.rotation_angle,))
        img_rotated = skimage.transform.rotate(img, self.rotation_angle, *self.args, **self.kwargs).astype(img.dtype)
        logger.info("Applying %0.02f rotation to mask via skimage.transform.rotate" % (self.rotation_angle,))
        mask_rotated = skimage.transform.rotate(mask, self.rotation_angle, *self.args, **self.kwargs)
        mask_rotated = np.logical_not(np.isclose(mask_rotated, np.zeros(mask.shape), atol=.0001))

        return GenericImageEntity(img_rotated, mask_rotated)


class RandomRotateXForm(ImageTransform):
    """Implements a rotation of a random amount of degrees.

    """
    def __init__(self, angle_choices: Sequence[float] = None, angle_sampler_prob: Sequence[float] = None,
                 rotator_kwargs: Dict = None) -> None:
        """
        Creates a random rotator Transform object
        :param angle_choices: An Sequence object of floats which represent the possible angles
                              from which the sampler can choose from
        :param angle_sampler_kwargs: any keyword arguments to pass to the sampler
        """
        if angle_choices is None:
            self.angle_choices = [0, 90, 180, 270]
        else:
            self.angle_choices = angle_choices
        if rotator_kwargs is None:
            self.rotator_kwargs = {'preserve_range': True}
        else:
            self.rotator_kwargs = rotator_kwargs
        self.angle_sampler_prob = angle_sampler_prob

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Samples from the possible angles according to the sampler specification and then applies that
        rotation to the input object
        :param input_obj: Entity to be randomly rotated
        :param random_state_obj: a random state used to maintain reproducibility through transformations
        :return: the transformed Entity
        """
        rotation_angle = random_state_obj.choice(self.angle_choices, p=self.angle_sampler_prob)

        logger.info("Sampled %0.02f rotation from RandomState" % (rotation_angle,))
        rotator = RotateXForm(rotation_angle, kwargs=self.rotator_kwargs)

        return rotator.do(input_obj, random_state_obj)
