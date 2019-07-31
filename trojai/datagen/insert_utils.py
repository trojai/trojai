from trojai.datagen.config import ValidInsertLocationsConfig

from typing import Callable, Sequence, Any

import numpy as np

import logging
logger = logging.getLogger(__name__)


def pattern_fit(chan_img: np.ndarray, chan_pattern: np.ndarray, chan_location: Sequence[Any]) -> bool:
    """
    Returns True if the pattern at the desired location can fit into the image channel without wrap, and False otherwise

    :param chan_img: a numpy.ndarray of shape (nrows, ncols) which represents an image channel
    :param chan_pattern: a numpy.ndarray of shape (prows, pcols) which represents a channel of the pattern
    :param chan_location: a Sequence of length 2, which contains the x/y coordinate of the top left corner of the
            pattern to be inserted for this specific channel
    :return: True/False depending on whether the pattern will fit into the image
    """

    p_rows, p_cols = chan_pattern.shape
    r, c = chan_location
    i_rows, i_cols = chan_img.shape

    if (r + p_rows) > i_rows or (c + p_cols) > i_cols:
        return False
    return True


def _score_avg_intensity(img_subset: np.ndarray) -> np.ndarray:
    return np.mean(img_subset)


def _check_corners(img_subset: np.ndarray) -> np.ndarray:
    r, c = img_subset.shape
    return img_subset[0][0] or img_subset[0][c - 1] or img_subset[r - 1][0] or img_subset[r - 1][c - 1]


def valid_locations(img: np.ndarray, pattern: np.ndarray, algo_config: ValidInsertLocationsConfig,
                    protect_wrap: bool = True, allow_overlap: bool = False) -> np.ndarray:
    """
    Returns a list of locations per channel which the pattern can be inserted
    into the img_channel with an overlap algorithm dicated by the appropriate
    inputs

    :param img: a numpy.ndarray which represents the image of shape:
           (nrows, ncols, nchans)
    :param pattern: the pattern to be inserted into the image of shape:
           (prows, pcols, nchans)
    :param protect_wrap: if True, ensures that pattern to be inserted can fit
           without wrapping and raises an Exception otherwise
    :param allow_overlap: if True, then valid locations include locations which
           would overlap any existing images
    :param algo_config: The provided configuration object specifying the algorithm to use and necessary parameters

    :return: A boolean mask of the same shape as the input image, with True
             indicating that that pixel is a valid location for placement of
             the specified pattern
    """
    num_chans = img.shape[2]

    # broadcast the allow_overlap variable if necessary
    if isinstance(allow_overlap, bool):
        allow_overlap = [allow_overlap] * num_chans

    if pattern.shape[2] != num_chans:
        # force user to broadcast the pattern as necessary
        msg = "The # of channels in the pattern does not match the # of channels in the image!"
        logger.error(msg)
        raise ValueError(msg)

    # TODO: look for vectorization opportunities
    output_mask = np.zeros(img.shape, dtype=bool)
    for chan_idx in range(num_chans):
        chan_img = img[:, :, chan_idx]
        chan_pattern = pattern[:, :, chan_idx]
        i_rows, i_cols = chan_img.shape
        p_rows, p_cols = chan_pattern.shape

        if allow_overlap[chan_idx]:
            output_mask[0:i_rows - p_rows + 1,
                        0:i_cols - p_cols + 1,
                        chan_idx] = True
        else:
            if protect_wrap:
                mask = np.full(chan_img.shape, True)
                # remove boundaries from valid locations
                mask[i_rows - p_rows + 1:i_rows, :] = False
                mask[:, i_cols - p_cols + 1:i_cols] = False

                if algo_config.algorithm == 'corner_check':
                    logger.info("Computing valid locations according to corner_check algorithm")
                elif algo_config.algorithm == 'threshold':
                    logger.info("Computing valid locations according to threshold algorithm")
                    mask = (chan_img <= algo_config.min_val)

                for i in range(i_rows):
                    for j in range(i_cols):
                        if mask[i][j]:
                            if algo_config.scorer(i, j, p_rows, p_cols, chan_img):
                                mask[i][j] = False
                output_mask[:, :, chan_idx] = mask

    return output_mask
