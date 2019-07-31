import cv2
import numpy as np
from typing import Tuple, Optional

import logging
logger = logging.getLogger(__name__)

"""
Contains general utilities for dealing with channel formats
"""


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert given grayscale image to RGB
    :param img: 1-channel grayscale image
    :return: image converted to RGB
    """
    if not (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
        raise TypeError("input image must be in either shape (rows, cols) or (rows, cols, 1)!")

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def rgba_to_rgb(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Split given 4-channel RGBA array into a 3-channel RGB array and a 1-channel alpha array
    :param img: given image to split, must be 3-channel or 4-channel
    :return: the first three channels of data as a 3-channel RGB image and the fourth channel of img as either a
    1-channel alpha array, or None if img has only 3 channels
    """
    if len(img.shape) != 3:
        raise TypeError('Input image must be in shape (rows, cols, channels)!')

    if img.shape[2] == 3:
        return img, None
    elif img.shape[2] == 4:
        r_ch, g_ch, b_ch, alpha_ch = cv2.split(img)
        return cv2.merge((r_ch, g_ch, b_ch)), alpha_ch
    else:
        raise TypeError("rgba_to_rgb expects a 3-channel or 4-channel input image,"
                        "%d-channel input image was detected!" % img.shape[2])


def rgb_to_rgba(img, alpha_ch: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Converts given image to RGBA, with optionally provided alpha_ch as its alpha channel
    :param img: 3-channel RGB image or 4-channel RGBA image
    :param alpha_ch: 1-channel array to be used as alpha value (optional),
    if img is RGBA this value is ignored
    :return: if img is 4-channel it is returned unmodified, if img is 3-channel this will return a new RGBA image with
    img as its RGB channels and either alpha_ch as its alpha channel if provided or a fully opaque alpha channel
    (max value for its datatype)
    """
    if len(img.shape) != 3:
        raise TypeError('Input image must be in shape (rows, cols, channels)!')

    if img.shape[2] == 3:
        if alpha_ch is None:
            return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        else:
            if len(alpha_ch.shape) == 2:
                return cv2.merge((img, alpha_ch))
            else:
                raise TypeError("input alpha channel to be must in shape (rows, cols)!")
    elif img.shape[2] == 4:
        return img
    else:
        raise TypeError("rgb_to_rgba expects a 3-channel or 4-channel input image, "
                        "%d-channel input image was detected!" % img.shape[2])


def normalization_to_rgb(img: np.ndarray, normalize: bool, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Guard for input to RGB only xforms
    :param img: input image with variable number of channels
    :param normalize: whether to attempt to convert img from original channel format to 3-channel RGB
    :param name: name of calling xform
    :return: a 3-channel RGB array converted from img,
    additional conversions can be added below, currently only RGBA to RGB is implemented
    """
    if len(img.shape) != 3:
        raise TypeError('Input image must be in shape (rows, cols, channels)!')

    original_n_chan = img.shape[2]
    if original_n_chan == 3:
        return img, None

    if normalize:
        if original_n_chan == 4:
            return rgba_to_rgb(img)
        else:
            raise TypeError("No conversion for %d-channel to 3-channel images is implemented!" % original_n_chan)
    else:
        raise TypeError("%s is an RGB-only transform, %d-channel input was detected!" % (name, original_n_chan,))


def normalization_from_rgb(rgb_img: np.ndarray, alpha_ch: Optional[np.ndarray], normalize: bool, original_n_chan: int,
                           name: str) -> np.ndarray:
    """
    Guard for output from rgb-only xforms
    :param rgb_img: 3-channel RGB image result from calling xform
    :param alpha_ch: alpha channel extracted at beginning of calling xform or None
    :param normalize: whether to convert rgb_img back to its original channel format
    :param original_n_chan: number of channels in its original channel format
    :param name: name of calling xform
    :return: if normalize is True the image corresponding to rgb_img converted to its original channel format,
    otherwise rgb_img unmodified,
    additional conversions can be added below, currently only RGB to RGBA is implemented
    """
    if len(rgb_img.shape) != 3:
        raise TypeError('Input image must be in shape (rows, cols, channels)!')

    if rgb_img.shape[2] != 3:
        raise TypeError("Input image must be in 3-channel format!")

    if original_n_chan == 3:
        return rgb_img

    if normalize:
        if original_n_chan == 4:
            return rgb_to_rgba(rgb_img, alpha_ch)
        else:
            raise TypeError("No conversion for 3-channel to %d-channel images is implemented!" % original_n_chan)
    else:
        logger.warning("%s is converting %d-channel images to RGB images!" % (name, original_n_chan,))
        return rgb_img
