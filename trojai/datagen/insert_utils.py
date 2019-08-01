from trojai.datagen.config import ValidInsertLocationsConfig

from typing import Callable, Sequence, Any

import numpy as np

import logging
logger = logging.getLogger(__name__)

def edge_pixel(local_img: np.ndarray) -> bool:
    for i in range(0, 3):
        for j in range(0, 3):
            #if i == 1 and j == 1:
            #    continue
            if local_img[i][j] == 0:
                return True
    return False


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


    if np.sum(chan_img[r:r + p_rows, c:c + p_cols]) > 0.0:
        print(np.sum(chan_img[r:r + p_rows, c:c + p_cols]))
        print(chan_img[r:r + p_rows, c:c + p_cols])
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
                mask = (chan_img <= algo_config.min_val)

                if algo_config.algorithm == 'corner_check':
                    logger.info("Computing valid locations according to corner_check algorithm")
                elif algo_config.algorithm == 'threshold':
                    logger.info("Computing valid locations according to threshold algorithm")

                # remove boundaries from valid locations
                mask[i_rows - p_rows + 1:i_rows, :] = False
                mask[:, i_cols - p_cols + 1:i_cols] = False

                if algo_config.algorithm == 'edge_tracing':
                    # move along edge of image filling in invalid locations
                    start = ()
                    for j in range(i_cols): # get start point on edge
                        if chan_img[i_rows // 2][j]:
                            start = (i_rows // 2, j)
                            break
                    if start == ():
                        print("Did not find object!")
                        raise ValueError("fasfasf")

                    start_i, start_j = start
                    # invalidate for start square
                    for i in range(start_i - p_rows + 1, start_i + p_rows):
                        for j in range(start_j - p_cols + 1, start_j + p_cols):
                            if 0 <= i < i_rows and 0 <= j < i_rows and mask[i][j]:
                                mask[i][j] = False

                    curr_i, curr_j = start_i, start_j
                    actions = []
                    moves = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
                    seen = {(curr_i, curr_j)}
                    while 1:
                        # get next square to check, clockwise
                        for move_i, move_j in moves:
                            # if part of image, on edge of image, and not visited already
                            if chan_img[curr_i + move_i][curr_j + move_j] > algo_config.min_val and \
                                edge_pixel(chan_img[curr_i + move_i - 1: curr_i + move_i + 2, curr_j + move_j - 1: curr_j + move_j + 2]) and \
                                (curr_i + move_i, curr_j + move_j) not in seen:
                                curr_i += move_i
                                curr_j += move_j
                                seen.add((curr_i, curr_j)) # update seen pixels
                                print(curr_i, curr_j)
                            else:
                                continue

                            # update invalidation based on last move
                            if move_i != 0:
                                for inv_j in range(curr_j - p_cols + 1, curr_j + p_cols):
                                    if 0 <= inv_j < i_cols:
                                        mask[curr_i + move_i * (p_rows - 1)][inv_j] = False
                            if move_j != 0:
                                for inv_i in range(curr_i - p_rows + 1, curr_i + p_rows):
                                    if 0 <= inv_i < i_rows:
                                        mask[inv_i][curr_j + move_j * (p_cols - 1)] = False

                        if curr_i == start_i and curr_j == start_j:  # if lap around image completed
                            break

                    output_mask[:, :, chan_idx] = mask

            else:
                msg = "Wrapping for trigger insertion has not been implemented yet!"
                logger.error(msg)
                raise ValueError(msg)

    return output_mask

    """
    for i in range(i_rows - p_rows + 1):
        for j in range(i_cols - p_cols + 1):
            if not mask[i][j]:
                # mark all four corners invalid
                top = i - p_rows + 1
                bottom = i + p_rows - 1
                left = j - p_cols + 1
                right = j + p_cols - 1
                if top >= 0 and left >= 0:
                    mask[top][left] = False
                if top >= 0 and right < i_cols:
                    mask[top][right] = False
                if bottom < i_rows and left >= 0:
                    mask[bottom][left] = False
                if bottom < i_rows and right < i_cols:
                    mask[bottom][right] = False
                mask[i]
    """


"""
for i in range(i_rows - p_rows + 1):
        for j in range(i_cols - p_cols + 1):
            if mask[i][j]:
                #if algo_config.scorer(i, j, p_rows, p_cols, chan_img):
                condition = False
                if algo_config.algorithm == 'corner_check':
                    condition = not ((not np.sum(chan_img[i:i + p_rows, j])) and
                                     (not np.sum(chan_img[i:i + p_rows, j + p_cols - 1])) and
                                     (not np.sum(chan_img[i, j:j + p_cols])) and
                                     (not np.sum(chan_img[i + p_rows - 1, j:j + p_cols])))
                    condition = chan_img[i][j] or \
                                chan_img[i][j + p_cols - 1] or \
                                chan_img[i + p_rows - 1][j] or \
                                chan_img[i + p_rows - 1][j + p_cols - 1]
                    #condition = algo_config.scorer(i, j, p_rows, p_cols, chan_img)
                elif algo_config.algorithm == 'threshold':
                    condition = np.mean(chan_img[i:i + p_rows, j:j + p_cols]) > algo_config.min_val
                if condition:
                    mask[i][j] = False
    """
