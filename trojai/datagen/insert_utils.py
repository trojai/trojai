import logging
from typing import Callable, Sequence, Any

import numpy as np
from joblib import Parallel, delayed

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


def pattern_overlap(chan_img: np.ndarray, chan_pattern: np.ndarray, chan_location: np.ndarray,
                    score_function: Callable[[np.ndarray], np.ndarray] = _score_avg_intensity,
                    algo: str = 'threshold', algo_config: dict = None) -> bool:
    """
    Returns True if the pattern overlaps part of the image

    :param chan_img: a numpy.ndarray of shape (nrows, ncols) which represents
           an image channel
    :param chan_pattern: a numpy.ndarray of shape (prows, pcols) which
           represents a channel of the pattern
    :param chan_location: a tuple or list of length 2, which contains the x/y
           coordinate of the top left corner of the pattern to be inserted for
           this specific channel
    :param score_function: a function handle to a function which accepts a 2-D
           image and produce some scalar value (score)
    :param algo: a string indicating which algorithm to use to determine if
           pattern overlaps image. Possibilities include:
            - threshold: simple algorithm that checks if a threshold is exceeded
                         over the size of the pattern
    :param algo_config: a dictionary containing the necessary hyperparameters
           for the overlap detection algorithm

    :return: True/False depending on whether the pattern overlaps or not, based
             on the input arguments
    """
    if algo_config is None:
        algo_config = dict(min_val=5)
    p_rows, p_cols = chan_pattern.shape
    r, c = chan_location
    img_subset = chan_img[r:r + p_rows, c:c + p_cols]
    img_subset_score = score_function(img_subset)

    if algo == 'threshold':
        if img_subset_score > algo_config['min_val']:
            return True
    else:
        msg = "Specified overlap algorithm not yet implemented!"
        logger.error(msg)
        raise ValueError(msg)

    return False


def valid_locations(img: np.ndarray, pattern: np.ndarray, protect_wrap: bool = True, allow_overlap: bool = False,
                    algo: str = 'threshold', algo_config: dict = None, njobs: int = 1) -> np.ndarray:
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
    :param algo: The algorithm to determine overlaps
           (only used if allow_overlap is False)
    :param algo_config: The necessary configuration for the specified algorithm
    :param njobs: The # of parallel processes to use. -1 means use all available

    :return: A boolean mask of the same shape as the input image, with True
             indicating that that pixel is a valid location for placement of
             the specified pattern
    """
    if algo_config is None:
        algo_config = dict(min_val=5)
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
            if algo == 'threshold' and protect_wrap:
                mask = (chan_img <= algo_config['min_val'])
                # # remove boundaries from valid locations
                mask[i_rows - p_rows + 1:i_rows, :] = False
                mask[:, i_cols - p_cols + 1:i_cols] = False

                # TODO: there is likely a better way to reduce the search-space
                #  even more - investigate
                # for every point in the mask, we see if pattern overlaps
                all_inds = np.arange(i_rows * i_cols)
                valid_inds = np.unravel_index(all_inds[mask.flatten()],
                                              (i_rows, i_cols))

                num_valid_inds = len(valid_inds[0])
                logger.info("Computing valid locations according to threshold algorithm")
                valid_loc_list = Parallel(n_jobs=njobs)(delayed(pattern_overlap)(chan_img=chan_img,
                                                                                 chan_pattern=chan_pattern,
                                                                                 chan_location=[valid_inds[0][ii],
                                                                                                valid_inds[1][ii]],
                                                                                 algo=algo, algo_config=algo_config)
                                                        for ii in range(num_valid_inds))
                # assign after compute: TODO - is there a cleaner way to assign?
                for ii in range(num_valid_inds):
                    x_idx = valid_inds[0][ii]
                    y_idx = valid_inds[1][ii]
                    output_mask[x_idx, y_idx, chan_idx] = not valid_loc_list[ii]
            else:
                msg = "Specified algorithm not yet implemented!"
                logger.error(msg)
                raise ValueError(msg)

    return output_mask
