from trojai.datagen.transform_interface import ImageTransform
from trojai.datagen.image_entity import ImageEntity, GenericImageEntity

from numpy.random import RandomState

import logging
logger = logging.getLogger(__name__)

"""
Defines several transformations related to color manipulation
"""


class GrayscaleRGBToRandomColorXForm(ImageTransform):
    """ Transformer to convert an RGB image to a random color image

    Converts each channel in a 3-channel image to a random color.
    Only pixels which exceed the defined threshold are modified.  This is done for each channel individually.
    """
    def __init__(self, thresh: float = 5) -> None:
        """
        Creates the grayscale-rgb to random color transform object
        :param thresh: threshold above which the sampled R/G/B value will be applied to the input Entity.
        """
        self.thresh = thresh

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual conversion
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: a np.random.RandomState object used to sample the colors and maintain reproducibility
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        if len(img.shape) != 3:
            raise ValueError("Image is not RGB channel - convert first!")

        img[img[:, :, 2].squeeze() > self.thresh, 2] = random_state_obj.choice(255)
        img[img[:, :, 1].squeeze() > self.thresh, 1] = random_state_obj.choice(255)
        img[img[:, :, 0].squeeze() > self.thresh, 0] = random_state_obj.choice(255)

        logger.info("Converted 3-channel Grayscale image to a random color")

        return GenericImageEntity(img, input_obj.get_mask())


class GrayscaleRGBToRandomGrayscaleColorXForm(ImageTransform):
    """
    Converts an RGB grayscale image to an RGB color image by only modifying one channel which is selected randomly,
    where the color gradient of the chosen channel is scaled similarly to the grayscale image
    """
    def __init__(self) -> None:
        """
        Initializes the grayscale-rgb to random color transform object
        """
        pass

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual specified transformation on the input Entity
        :param input_obj: the input object to be transformed
        :param random_state_obj: a np.random.RandomState object which performs the sampling of which channel to modify
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        if len(img.shape) != 3:
            raise ValueError("Image is not RGB channel - convert first!")

        channel = random_state_obj.choice(3) # choose which channel to modify
        # zero out the channels that we don't want color for to produce a
        # grayscale "like" color image
        for ii in range(3):
            if ii != channel:
                img[:, :, ii] = 0

        logger.info("Converted 3-channel Grayscale image with full color applied to channel=%d" % (channel,))

        return GenericImageEntity(img, input_obj.get_mask())
