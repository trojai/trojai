from typing import Dict

import math
import numpy as np
from PIL import Image, ImageEnhance, ImageStat

import logging
logger = logging.getLogger(__name__)

"""
Module contains various utilities to modify the lighting of an image.
"""


def find_image_exposure(img_input: np.ndarray) -> Dict:
    """
    Gets various metrics related to the image exposure, and returns them in a dictionary
    :param img_input: the input image, for which exposure metrics should be calculated
    :return:
    """
    n_chan = img_input.shape[2]
    img = Image.fromarray(img_input.astype(np.uint8))
    im = Image.fromarray(img_input.astype(np.uint8)).convert('LA')
    stat = ImageStat.Stat(im)
    # Average pixel brighness
    avg = stat.mean[0]
    # RMS pixel brighness
    rms = stat.rms[0]
    stat2 = ImageStat.Stat(img)

    # Consider the number of channels
    # background may have RGB while traffic sign has RGBA
    if n_chan == 3:
        # Average pixels preceived brightness
        r, g, b = stat2.mean
        avg_perceived = math.sqrt(
            0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
        # RMS pixels perceived brightness
        r, g, b = stat2.rms
        rms_perceived = math.sqrt(
            0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
        # exposure_info = [avg, rms, avg_perceived, rms_perceived]
    else:
        # Average pixels preceived brightness
        r, g, b, a = stat2.mean
        avg_perceived = math.sqrt(
            0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
        # RMS pixels perceived brightness
        r, g, b, a = stat2.rms
        rms_perceived = math.sqrt(
            0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
        # exposure_info = [avg, rms, avg_perceived, rms_perceived]

    exposure_info = dict(avg=avg, rms=rms, perceived_avg=avg_perceived, perceived_rms=rms_perceived)
    logger.debug("Computed image exposure info: %s" % (str(exposure_info),))
    return exposure_info


def adjust_brightness_mmavg(fg_img: np.ndarray, bg_img: np.ndarray, avrg_ratio: float = 11.01) -> np.ndarray:
    """
    Adjust's brightness by minimizing the margin based on the average of the two channel brightness variations.

    :param fg_img: The foreground image
    :param bg_img: The background image
    :param avrg_ratio: the ratio of brightness to maintain between foreground and background images according to
            the average criterion
    :return: the brightness adjust image
    """
    peak = Image.fromarray(fg_img.astype(np.uint8)).convert('LA')
    stat = ImageStat.Stat(peak)
    avrg = stat.mean[0]
    peak2 = Image.fromarray(fg_img.astype(np.uint8)).convert('RGBA')
    enhancer = ImageEnhance.Brightness(peak2)
    background_exposures = find_image_exposure(bg_img)
    margin = abs(avrg - float(background_exposures['avg']))
    brightness_avrg = margin / avrg_ratio
    avrg_bright = enhancer.enhance(brightness_avrg)
    logger.debug("Enhanced brightness by %0.02f according to the minimize average channel margin method" %
                (brightness_avrg,))
    return np.array(avrg_bright)


def adjust_brightness_mmrms(fg_img: np.ndarray, bg_img: np.ndarray, rms_ratio: float = 8.3) -> np.ndarray:
    """
    Adjust's brightness by minimizing the margin based on the RMS of the two channel brightness variations.

    :param fg_img: The foreground image
    :param bg_img: The background image
    :param rms_ratio: the ratio of brightness to maintain between foreground and background images according to
            the RMS criterion
    :return: the brightness adjust image
    """
    peak = Image.fromarray(fg_img.astype(np.uint8)).convert('LA')
    stat = ImageStat.Stat(peak)
    rms = stat.rms[0]
    peak2 = Image.fromarray(fg_img.astype(np.uint8)).convert('RGBA')
    enhancer = ImageEnhance.Brightness(peak2)
    background_exposures = find_image_exposure(bg_img)
    margin = abs(rms - float(background_exposures['rms']))
    brightness_avrg = margin / rms_ratio
    rms_bright = enhancer.enhance(brightness_avrg)
    logger.debug("Enhanced brightness by %0.02f according to the minimize RMS channel margin method" %
                (brightness_avrg,))
    return np.array(rms_bright)


def adjust_brightness_mmpavg(fg_img: np.ndarray, bg_img: np.ndarray, percieved_avrg_ratio: float = 3.85) -> np.ndarray:
    """
    Adjust's brightness by minimizing the margin based on the perceived average of the two channel brightness variations
    REFERENCE FOR ALGORITHM USED: http://alienryderflex.com/hsp.html

    :param fg_img: The foreground image
    :param bg_img: The background image
    :param percieved_avrg_ratio: the ratio of brightness to maintain between foreground and background images according
            to the perceived average criterion
    :return: the brightness adjust image
    """
    peak2 = Image.fromarray(fg_img.astype(np.uint8)).convert('RGBA')
    stat2 = ImageStat.Stat(peak2)
    r, g, b, a = stat2.mean
    avrg_perceived = math.sqrt(
        0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

    enhancer = ImageEnhance.Brightness(peak2)
    background_exposures = find_image_exposure(bg_img)
    margin = abs(avrg_perceived - float(background_exposures['perceived_avg']))
    brightness_avrg = margin / percieved_avrg_ratio
    rms_bright = enhancer.enhance(brightness_avrg)
    logger.debug("Enhanced brightness by %0.02f according to the perceived average difference between channels method" %
                (brightness_avrg,))
    return np.array(rms_bright)


def adjust_brightness_mmprms(fg_img: np.ndarray, bg_img: np.ndarray, percieved_rms_ratio: float = 35.6) -> np.ndarray:
    """
    Adjust's brightness by minimizing the margin based on the perceived RMS of the two channel brightness variations
    REFERENCE FOR ALGORITHM USED: http://alienryderflex.com/hsp.html

    :param fg_img: The foreground image
    :param bg_img: The background image
    :param percieved_rms_ratio: the ratio of brightness to maintain between foreground and background images according
            to the perceived RMS criterion
    :return: the brightness adjust image
    """
    peak2 = Image.fromarray(fg_img.astype(np.uint8)).convert('RGBA')
    stat2 = ImageStat.Stat(peak2)
    r, g, b, a = stat2.mean
    avrg_perceived = math.sqrt(
        0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

    enhancer = ImageEnhance.Brightness(peak2)
    background_exposures = find_image_exposure(bg_img)
    margin = abs(avrg_perceived - float(background_exposures['perceived_rms']))
    brightness_avrg = margin / percieved_rms_ratio
    rms_bright = enhancer.enhance(brightness_avrg)
    logger.debug("Enhanced brightness by %0.02f according to the perceived RMS difference between channels method" %
                (brightness_avrg,))
    return np.array(rms_bright)
