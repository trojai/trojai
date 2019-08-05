import time
from collections import deque

from trojai.datagen.config import InsertAtRandomLocationConfig

from typing import Callable, Sequence, Any

import numpy as np
from scipy.ndimage import filters

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

    if not valid_location(chan_img, chan_pattern, chan_location):
        return False

    return True


def valid_location(chan_img: np.ndarray, chan_pattern: np.ndarray, chan_location: Sequence[Any]) -> bool:
    """
    Returns False if the pattern intersects with the given image for top-left corner location

    :param chan_img: a numpy.ndarray of shape (nrows, ncols) which represents an image channel
    :param chan_pattern: a numpy.ndarray of shape (prows, pcols) which represents a channel of the pattern
    :param chan_location: a Sequence of length 2, which contains the x/y coordinate of the top left corner of the
            pattern to be inserted for this specific channel
    :return: True/False depending on whether the location is valid for the given image and pattern
    """

    p_rows, p_cols = chan_pattern.shape
    r, c = chan_location

    if np.logical_or.reduce(chan_img[r:r + p_rows, c:c + p_cols], axis=None):
        return False

    return True


def _score_avg_intensity(img_subset: np.ndarray) -> np.ndarray:
    return np.mean(img_subset)


def valid_locations(img: np.ndarray, pattern: np.ndarray, algo_config: InsertAtRandomLocationConfig,
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
                mask = (chan_img <= algo_config.min_val)

                if algo_config.algorithm == 'corner_check':
                    logger.info("Computing valid locations according to corner_check algorithm")
                elif algo_config.algorithm == 'threshold':
                    logger.info("Computing valid locations according to threshold algorithm")

                img_mask = np.logical_not(mask)  # True if image present, false if not

                # remove boundaries from valid locations
                mask[i_rows - p_rows + 1:i_rows, :] = False
                mask[:, i_cols - p_cols + 1:i_cols] = False

                # TODO: implement moves of variable length i.e. split image into edges instead of single steps
                if algo_config.algorithm == 'edge_tracing':
                    start = time.time()
                    # generate all edge pixels
                    edges = np.nonzero(
                        np.logical_and(
                            np.logical_xor(
                                filters.maximum_filter(img_mask, 3, mode='constant', cval=0.0),
                                filters.minimum_filter(img_mask, 3, mode='constant', cval=0.0)),
                            img_mask))
                    edges = zip(edges[0], edges[1])
                    remaining = set(edges)
                    # search until all edges have been visited
                    while len(remaining) != 0:
                        start_i, start_j = remaining.pop()

                        # invalidate relevant pixels for start square
                        top_boundary = max(0, start_i - p_rows + 1)
                        left_boundary = max(0, start_j - p_cols + 1)
                        mask[top_boundary:start_i + 1,
                             left_boundary: start_j + 1] = False

                        # all single pixels steps to take from a pixel
                        diag_moves = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
                        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
                        moves += diag_moves

                        actions = list()
                        actions.append((start_i, start_j, 0, 0))
                        while len(actions) != 0:
                            # where you are, what move you took to get there
                            curr_i, curr_j, action_i, action_j = actions.pop()

                            # truncate when near top or left boundary
                            top_index = max(0, curr_i - p_rows + 1)
                            left_index = max(0, curr_j - p_cols + 1)

                            # update invalidation based on last move, check for image mask assumes convexity of image,
                            # i.e. if both corners present assumes all pixels in between are filled in
                            if action_i < 0:
                                # update top border
                                mask[top_index:top_index - action_i, left_index:curr_j + 1] = False

                            elif action_i > 0:
                                # update bottom border
                                mask[curr_i - action_i + 1:curr_i + 1, left_index:curr_j + 1] = False

                            if action_j < 0:
                                # update left border
                                mask[top_index:curr_i + 1, left_index:left_index - action_j] = False

                            elif action_j > 0:
                                # update right border
                                mask[top_index:curr_i + 1, curr_j - action_j + 1:curr_j + 1] = False

                            # get next pixels to check, from neighbors of curr pixel + action
                            found = False
                            for dir_i, dir_j in moves:
                                new_i, new_j = curr_i, curr_j
                                # make as large a move as possible in dir_x, dir_y,
                                # adding to its size until you hit a non-edge pixel or array edge
                                while 0 <= new_i + dir_i < i_rows and 0 <= new_j + dir_j < i_cols and \
                                        (new_i + dir_i, new_j + dir_j) in remaining:
                                    found = True
                                    # update seen pixels
                                    new_i += dir_i
                                    new_j += dir_j
                                    remaining.remove((new_i, new_j))
                                    # only single moves for diagonal directions
                                    if (dir_i, dir_j) in diag_moves:
                                        break
                                if found:
                                    # next location/action to visit
                                    actions.append((new_i, new_j, new_i - curr_i, new_j - curr_j))
                                    break

                        output_mask[:, :, chan_idx] = mask
                        print(img_mask.astype(np.uint8))
                        print(mask.astype(np.uint8))

                elif algo_config.algorithm == 'brute_force':
                    edges = np.nonzero(
                        np.logical_and(
                            np.logical_xor(
                                filters.maximum_filter(img_mask, 3, mode='constant', cval=0.0),
                                filters.minimum_filter(img_mask, 3, mode='constant', cval=0.0)),
                            img_mask))
                    edges = zip(edges[0], edges[1])
                    for i, j in edges:
                        mask[max(0, i - p_rows + 1):i + 1, max(0, j - p_cols + 1):j + 1] = False
                    output_mask[:, :, chan_idx] = mask

                elif algo_config.algorithm == 'kd_fill':
                    edges = np.nonzero(
                        np.logical_and(
                            np.logical_xor(
                                filters.maximum_filter(img_mask, 3, mode='constant', cval=0.0),
                                filters.minimum_filter(img_mask, 3, mode='constant', cval=0.0)
                            ),
                            img_mask
                        )
                    )
                    edges = list(zip(edges[0], edges[1]))

            else:
                msg = "Wrapping for trigger insertion has not been implemented yet!"
                logger.error(msg)
                raise ValueError(msg)

    return output_mask
