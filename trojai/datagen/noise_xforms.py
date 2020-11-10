import cv2
import numpy as np
from numpy.random import RandomState

from trojai.datagen.transform_interface import ImageTransform
from trojai.datagen.image_entity import ImageEntity, GenericImageEntity

import logging
logger = logging.getLogger(__name__)

"""
Defines several transformations related to color manipulation
"""


class PoissonNoiseXForm(ImageTransform):
    """
    Inserts Poisson noise into the image object
    """
    def __init__(self, exponent_base: float = 2.05) -> None:
        """
        Initializes the Poisson Noise Inserter Object
        :param exponent_base:
        """
        self.exponent_base = exponent_base

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual noise insertion
        :param input_obj: the input Entity to which noise is to be added
        :param random_state_obj: a RandomState object used to sample the noise
        :return: the transformed (noise added) Entity
        """
        img = input_obj.get_data()
        vals = len(np.unique(img))
        vals = self.exponent_base ** np.ceil(np.log2(vals))
        noisy = random_state_obj.poisson(img * vals) / float(vals)

        logger.debug("Added Poisson Noise to Image")
        return GenericImageEntity(noisy, input_obj.get_mask())


class GaussianNoiseXForm(ImageTransform):
    """
    Inserts Gaussian noise into the image object
    """
    def __init__(self, mean: float = 0, var: float = 0.5) -> None:
        """
        Initializes the Gaussian Noise Inserter Object
        :param mean: noise mean value
        :param var: noise standard deviation
        :param seed: random seed value
        """
        self.mean = mean
        self.var = var

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual noise insertion
        :param input_obj: the input Entity to which noise is to be added
        :param random_state_obj: a RandomState object used to sample the noise
        :return: the transformed (noise added) Entity
        """
        img = input_obj.get_data()
        row, col, ch = img.shape
        sigma = self.var ** 0.5
        gaussian_noise = random_state_obj.normal(self.mean, sigma, (row, col, ch))
        gaussian_noise = gaussian_noise.reshape(row, col, ch)
        noisy = img + gaussian_noise

        logger.debug("Added Gaussian Noise to Image")
        return GenericImageEntity(noisy, input_obj.get_mask())


class GaussianBlurXForm(ImageTransform):
    """
    Performs Gaussian blurring of an Entity
    """
    def __init__(self, ksize: int = 5, sigmaX: float = 0, sigmaY: float = 0) -> None:
        """
        Initializes the Gaussian Blur Object
        :param ksize: kernel size
        :param sigmaX: x sigma value (leave as 0 to define based on kernel size)
        :param sigmaY: y sigma value (leave as 0 to define based on kernel size)
        """
        if ksize%2 == 0:
            raise ValueError("Kernel size must be odd!")
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual blurring operation
        :param input_obj: the input Entity to which noise is to be added
        :param random_state_obj: ignored
        :return: the transformed (noise added) Entity
        """
        img = input_obj.get_data()
        blurred = cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigmaX, self.sigmaY)

        logger.debug("Added Gaussian Blur w/ Kernel Size=%d to Image" % (self.ksize,))
        return GenericImageEntity(blurred, input_obj.get_mask())


class RandomGaussianBlurXForm(ImageTransform):
    """
    Performs Gaussian blurring of an Entity
    """
    def __init__(self, ksize_min: int = 0, ksize_max: int = 5, sigmaX: float = 0, sigmaY: float = 0) -> None:
        """
        Initializes the Gaussian Blur Object
        :param ksize_min: minimum kernel size to select and apply
        :param ksize_max: maximum kernel size to select and apply
        :param sigmaX: x sigma value (leave as 0 to define based on kernel size)
        :param sigmaY: y sigma value (leave as 0 to define based on kernel size)
        """

        self.ksize_min = int(max(ksize_min - 1, 0) / 2)
        self.ksize_max = int(max(ksize_max - 1, 0) / 2)
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual blurring operation
        :param input_obj: the input Entity to which noise is to be added
        :param random_state_obj: ignored
        :return: the transformed (noise added) Entity
        """

        # select a kernel size
        ksize = int(random_state_obj.randint(int(self.ksize_min), int(self.ksize_max)))
        ksize = int(ksize * 2 + 1)
        if ksize <= 1:
            return input_obj
        else:
            img = input_obj.get_data()
            blurred = cv2.GaussianBlur(img, (ksize, ksize), self.sigmaX, self.sigmaY)

            logger.debug("Added Gaussian Blur w/ Kernel Size=%d to Image" % (ksize,))
            return GenericImageEntity(blurred, input_obj.get_mask())