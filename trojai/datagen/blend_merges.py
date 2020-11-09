from trojai.datagen.merge_interface import ImageMerge
from trojai.datagen.image_entity import ImageEntity, GenericImageEntity
import trojai.datagen.lighting_utils as lighting_utils

from typing import Callable
import blend_modes
import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image

import logging
logger = logging.getLogger(__name__)


"""
Module which defines several blend style merge operations.
"""


class GrainMerge(ImageMerge):
    """ Performs a "grain" merge, similar to grain-merge in PhotoShop.

    See here for details:
    https://pythonhosted.org/blend_modes/blend_modes.html?highlight=grain_merge#blend_modes.blend_modes.grain_merge
    """
    def __init__(self, opacity: float = 1) -> None:
        """
        Creates the GrainMerge object
        :param opacity: Desired opacity of layer for blending
        """
        self.opacity = opacity

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual blend operation on an input and pattern
        :param img_obj: image to be blended upon
        :param pattern_obj: the layer to be blended with the image
        :param random_state_obj: ignored
        :return: the merged object
        """
        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern = pattern_obj.get_data()
        pattern_mask = pattern_obj.get_mask()
        logger.debug("Grain Merging img w/ shape=%s and pattern w/ shape=%s with opacity=%0.02f",
                    (str(img.shape), str(pattern.shape), self.opacity))
        img_out = blend_modes.grain_merge(img.astype(float), pattern.astype(float), self.opacity)
        mask_out = img_mask & pattern_mask
        return GenericImageEntity(img_out, mask_out)


class AddMerge(ImageMerge):
    """ Merge objects by element wise addition

    """
    def __init__(self) -> None:
        """
        Create the AddMerge object
        """
        pass

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual Merge operation on the input and pattern
        :param img_obj: image to be added
        :param pattern_obj: pattern to be added
        :param random_state_obj: ignored
        :return: the merged object
        """
        logger.debug("Add Merging img w/ shape=%s and pattern w/ shape=%s",
                    (str(img_obj.get_data().shape), str(pattern_obj.get_data().shape)))
        img_out = cv2.add(img_obj.get_data(), pattern_obj.get_data())
        # TODO: revisit whether this is the correct behavior for the mask
        mask_out = cv2.add(img_obj.get_mask(), pattern_obj.get_mask())
        return GenericImageEntity(img_out, mask_out)


class GrainMergePaste(ImageMerge):
    """ Class which implements a Grain Merge and Paste operation in serial on two Entities.

    See: https://pythonhosted.org/blend_modes/blend_modes.html?highlight=grain_merge#blend_modes.blend_modes.grain_merge
         https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.paste
    """
    def __init__(self, opacity: float = 1) -> None:
        """
        Creates a GrainMergeAndPaste object with a configurable opacity
        :param opacity: Desired opacity of layer for blending
        """
        self.opacity = opacity

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual blending operations
        :param img_obj: image to be blended upon
        :param pattern_obj: the layer to be blended with the image & pasted
        :param random_state_obj: ignored
        :return: the merged object
        """
        img = img_obj.get_data()
        img_mask = img_obj.get_mask()
        pattern = pattern_obj.get_data()
        pattern_mask = pattern_obj.get_mask()

        logger.debug("Grain Merging img w/ shape={} and pattern w/ shape={} with opacity={:0.02f}".format(str(img.shape), str(pattern.shape), self.opacity))

        img_r, img_c, _ = img.shape
        pat_r, pat_c, _ = pattern.shape
        if pat_r > img_r or pat_c > img_c:
            msg = "Pattern to be merged into image is larger than the image!"
            logger.error(msg)
            raise ValueError(msg)
        if pat_r < img_r or pat_c < img_c:
            # TODO: make this an option so that we have multiple resize options
            logger.debug("Resizing pattern to match image size with image background!")
            pattern_resized = img.copy()
            row_insert_idx = (img_r-pat_r)//2
            col_insert_idx = (img_c-pat_c)//2
            pattern_resized[row_insert_idx:row_insert_idx+pat_r, col_insert_idx:col_insert_idx+pat_c,:] = pattern
            pattern = pattern_resized.copy()
            pattern_mask_resized = np.zeros((img_r, img_c), dtype=bool)
            pattern_mask_resized[row_insert_idx:row_insert_idx+pat_r, col_insert_idx:col_insert_idx+pat_c] = pattern_mask
            pattern_mask = pattern_mask_resized.copy()

        blended_img = blend_modes.grain_merge(img.astype(float), pattern.astype(float), self.opacity)
        blended_img_raw = Image.fromarray(blended_img.astype(np.uint8))
        pattern_raw = Image.fromarray(pattern.astype(np.uint8))
        logger.debug("Pasting pattern into grain merged image")
        blended_img_raw.paste(pattern_raw, (0, 0), pattern_raw)
        final_img = np.array(blended_img_raw)

        # TODO: revisit whether this is the correct/intended behavior for the mask
        final_mask = img_mask & pattern_mask

        return GenericImageEntity(final_img, final_mask)


class BrightnessAdjustGrainMergePaste(ImageMerge):
    """ Class which implements a brightness adjustment before grain merging & pasting

    """
    def __init__(self, opacity: float = 1,
                 lighting_adjuster: Callable[[np.ndarray, np.ndarray], np.ndarray] =
                 lighting_utils.adjust_brightness_mmavg) -> None:
        """
        Create an instance of the brighness adjustment + grain merging + pasting object
        :param opacity: Desired opacity of layer for blending
        :param lighting_adjuster: A function handle for the lighting adjustment operation
        """
        self.opacity = opacity
        self.lighting_adjuster = lighting_adjuster

    def do(self, img_obj: ImageEntity, pattern_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the actual blending operations
        :param img_obj: image to be blended upon
        :param pattern_obj: the layer to be blended with the brightness adjusted & image  & pasted
        :param random_state_obj: ignored
        :return: the merged object
        """
        img = img_obj.get_data()
        pattern = pattern_obj.get_data()
        pattern_mask = pattern_obj.get_mask()

        # adjust brightness of pattern to match image
        logger.debug("Adjusting brightness according to:" + str(self.lighting_adjuster))
        pattern_adjusted = self.lighting_adjuster(pattern, img)
        pattern_adjusted_obj = GenericImageEntity(pattern_adjusted, pattern_mask)
        logger.debug("Performing GrainMergePaste with opacity = %0.02f", (self.opacity))
        merger = GrainMergePaste(self.opacity)
        return merger.do(img_obj, pattern_adjusted_obj, random_state_obj)
