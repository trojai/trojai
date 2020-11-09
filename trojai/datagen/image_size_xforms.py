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
        logger.debug("Resized image of shape=%s to shape=%s using %d interpolation" %
                    (str(img_obj.get_data().shape), str(self.new_size), self.interpolation))
        return GenericImageEntity(img_out, mask_out)


class RandomResize(Transform):
    """
    Resizes an Entity
    """
    def __init__(self, new_size_minimum: tuple = (200, 200), new_size_maximum: tuple = (300, 300), interpolation: int = cv2.INTER_CUBIC) -> None:
        """
        Initialize the resizer object
        :param new_size_minimum: a tuple of the minimum size in pixes for x and y dimensions
        :param new_size_maximum: a tuple of the maximum size in pixes for x and y dimensions
        :param interpolation: the interpolation method to resize the input Entity
        """
        self.new_size_minimum = new_size_minimum
        self.new_size_maximum = new_size_maximum
        self.interpolation = interpolation

    def do(self, img_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the resizing
        :param img_obj: The input object to be resized according the specified configuration
        :param random_state_obj: ignored
        :return: The resized object
        """
        # select a new size from within the range of calid new sizes
        y = random_state_obj.randint(self.new_size_minimum[0], self.new_size_maximum[0])
        x = random_state_obj.randint(self.new_size_minimum[1], self.new_size_maximum[1])
        new_size = (y, x)

        img_out = cv2.resize(img_obj.get_data(), new_size, interpolation=self.interpolation)
        mask_out = cv2.resize(img_obj.get_mask().astype(np.float32), new_size,
                              interpolation=self.interpolation).astype(bool)
        logger.debug("Resized image of shape=%s to shape=%s using %d interpolation" %
                    (str(img_obj.get_data().shape), str(new_size), self.interpolation))
        return GenericImageEntity(img_out, mask_out)


class RandomPadToSize(Transform):
    """
    Resizes an Entity
    """
    def __init__(self, new_size: tuple = (200, 200), mode: str = 'constant', pad_value: int = 0) -> None:
        """
        Initialize the resizer object
        :param new_size: a tuple of the size in pixes for x and y dimensions
        :param mode: what type of padding to use, supports numpy.pad options
        :param pad_value: the value to use when padding
        """
        self.new_size = new_size
        self.mode = mode
        self.pad_value = pad_value

    def do(self, img_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the resizing
        :param img_obj: The input object to be resized according the specified configuration
        :param random_state_obj: ignored
        :return: The resized object
        """

        input_shape = img_obj.get_data().shape

        if self.new_size[0] < input_shape[0] or self.new_size[1] < input_shape[1]:
            raise RuntimeError('Invalid pad new_size {} smaller than input image size {}'.format(self.new_size, input_shape))

        total_pad_y = self.new_size[0] - input_shape[0]
        total_pad_x = self.new_size[1] - input_shape[1]

        pre_pad_value_y = random_state_obj.randint(0, total_pad_y)
        pre_pad_value_x = random_state_obj.randint(0, total_pad_x)
        pad_values = (pre_pad_value_y, total_pad_y - pre_pad_value_y, pre_pad_value_x, total_pad_x - pre_pad_value_x)

        return Pad(pad_values, self.mode, self.pad_value).do(img_obj, random_state_obj)


class Pad(Transform):
    """
    Resizes an Entity
    """
    def __init__(self, pad_amounts: tuple = (0, 0, 0, 0), mode: str = 'constant', pad_value: int = 0) -> None:
        """
        Initialize the resizer object
        :param pad_amounts: a tuple of the pixel count o add to each side (y_pre, y_post, x_pre, x_post)
        :param mode: what type of padding to use, supports numpy.pad options
        :param pad_value: the value to use when padding
        """
        self.pad_amounts = pad_amounts
        self.mode = mode
        self.pad_value = pad_value

    def do(self, img_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the resizing
        :param img_obj: The input object to be resized according the specified configuration
        :param random_state_obj: ignored
        :return: The resized object
        """

        if np.count_nonzero(np.asarray(self.pad_amounts)) == 0:
            return img_obj

        img = img_obj.get_data()
        msk = img_obj.get_mask()
        kwargs = {}
        if self.mode == 'constant':
            kwargs = {'constant_values': self.pad_value}

        if len(img.shape) == 2:
                img_out = np.pad(img, pad_width=((self.pad_amounts[0], self.pad_amounts[1]), (self.pad_amounts[2], self.pad_amounts[3])), mode=self.mode, **kwargs)
        elif len(img.shape) == 3:
            img_out = np.pad(img, pad_width=((self.pad_amounts[0], self.pad_amounts[1]), (self.pad_amounts[2], self.pad_amounts[3]), (0, 0)), mode=self.mode, **kwargs)
        else:
            raise RuntimeError('Unexpected image shape: {}'.format(img.shape))

        if len(msk.shape) == 2:
            mask_out = np.pad(msk, pad_width=((self.pad_amounts[0], self.pad_amounts[1]), (self.pad_amounts[2], self.pad_amounts[3])), mode=self.mode, **kwargs)
        elif len(msk.shape) == 3:
            mask_out = np.pad(msk, pad_width=((self.pad_amounts[0], self.pad_amounts[1]), (self.pad_amounts[2], self.pad_amounts[3]), (0, 0)), mode=self.mode, **kwargs)
        else:
            raise RuntimeError('Unexpected mask shape: {}'.format(msk.shape))

        logger.debug("Padded image of shape=%s to shape=%s" %
                    (str(img_obj.get_data().shape), str(img.shape)))
        return GenericImageEntity(img_out, mask_out)


class RandomSubCrop(Transform):
    """
    Resizes an Entity
    """
    def __init__(self, new_size: tuple = (200, 200)) -> None:
        """
        Initialize the crop object
        :param new_size: a tuple of the size in pixels for x and y dimensions
        """
        self.new_size = new_size

    def do(self, img_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the resizing
        :param img_obj: The input object to be cropped according the specified configuration
        :param random_state_obj: ignored
        :return: The cropped object
        """
        img = img_obj.get_data()
        msk = img_obj.get_mask()

        if self.new_size[0] > img.shape[0]:
            raise RuntimeError('Invalid subcrop size: requested height {} is larger than source image height {}'.format(self.new_size[0], img.shape[0]))
        if self.new_size[1] > img.shape[1]:
            raise RuntimeError('Invalid subcrop size: requested width {} is larger than source image width {}'.format(self.new_size[1], img.shape[1]))

        if self.new_size[0] == img.shape[0] and self.new_size[1] == img.shape[1]:
            return img_obj

        y_st = np.random.randint(0, img.shape[0] - self.new_size[0])
        x_st = np.random.randint(0, img.shape[1] - self.new_size[1])

        img_out = img[y_st:y_st + self.new_size[0], x_st:x_st + self.new_size[1]]
        mask_out = msk[y_st:y_st + self.new_size[0], x_st:x_st + self.new_size[1]]

        logger.debug("Cropped source image size {} to output size {}".format(img_obj.get_data().shape, img_out.shape))
        return GenericImageEntity(img_out, mask_out)
