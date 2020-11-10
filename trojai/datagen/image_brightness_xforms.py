from trojai.datagen.transform_interface import ImageTransform
from trojai.datagen.image_entity import ImageEntity, GenericImageEntity

import numpy as np
from numpy.random import RandomState
from PIL import ImageEnhance, Image

import logging
logger = logging.getLogger(__name__)


class MMPRMSXForm(ImageTransform):
    """Implements brightness using PIL
    """
    def __init__(self, brightness: float = 1, kwargs: dict = None) -> None:
        """
        Create a scaler object
        :param scale_factor: the scaling amount
        :param kwargs: any keyword arguments to pass to skimge.transform.rescale
        """
        self.brightness = brightness
        if kwargs is None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs

    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Performs the scaling on an input Entity using skimage.transform.rescale
        :param input_obj: the input object to be scaled
        :param random_state_obj: ignored
        :return: the transformed Entity
        """
        img = input_obj.get_data()
        mask = input_obj.get_mask()

        logger.info("Applying %0.02f brightning of image" % (self.brightness,))
        enhancer = ImageEnhance.Sharpness(Image.fromarray(img))
        img_brightned = np.array(enhancer.enhance(self.brightness))

        return GenericImageEntity(img_brightned, mask)
