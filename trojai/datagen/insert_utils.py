from collections import deque

from trojai.datagen.config import InsertAtRandomLocationConfig

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
                    img_pixels = np.nonzero(img_mask)
                    # if no nonzero pixel values for image
                    if len(img_pixels[0]) == 0:
                        output_mask[0:i_rows - p_rows + 1,
                                    0:i_cols - p_cols + 1,
                                    chan_idx] = True
                    else:
                        # otherwise closest pixel to top-left
                        start_i, start_j = img_pixels[0][0], img_pixels[1][0]

                        # invalidate relevant pixels for start square
                        top_boundary = max(start_i - p_rows + 1, 0)
                        left_boundary = max(start_j - p_cols + 1, 0)
                        mask[top_boundary:start_i + 1,
                             left_boundary: start_j + 1] = False

                        # DFS along perimeter of image, invalidating a new row and/or column of pixels each time

                        # all single pixels steps to take from a pixel
                        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
                        # already visited edge pixels
                        seen = {(start_i, start_j)}
                        # used as stack of edge pixels to visit and the direction of move that brought it there
                        actions = list()
                        actions.append((start_i, start_j, 0, 0))
                        while len(actions) != 0:
                            # where you are, what move you took to get there
                            curr_i, curr_j, action_i, action_j = actions.pop()

                            # truncate when near top or left boundary
                            top_index = max(curr_i - p_rows + 1, 0)
                            left_index = max(curr_j - p_cols + 1, 0)

                            # update invalidation based on last move, check for image mask assumes convexity of image,
                            # i.e. if both corners present assumes all pixels in between are filled in
                            if action_i == -1:
                                # update top border
                                mask[top_index, left_index:curr_j + 1] = False

                            elif action_i == 1:
                                # update bottom border
                                mask[curr_i, left_index:curr_j + 1] = False

                            if action_j == -1:
                                # update left border
                                mask[top_index:curr_i + 1, left_index] = False

                            elif action_j == 1:
                                # update right border
                                mask[top_index:curr_i + 1, curr_j] = False

                            # get next pixels to check, from neighbors of curr pixel + action
                            for move_i, move_j in moves:
                                new_i = curr_i + move_i
                                new_j = curr_j + move_j
                                # border checking
                                if new_i < 0 or new_i >= i_rows or new_j < 0 or new_j >= i_cols:
                                    continue

                                # 3 x 3 square around possible move
                                local_img_mask = img_mask[max(0, new_i - 1): min(i_rows, new_i + 2),
                                                          max(0, new_j - 1): min(i_cols, new_j + 2)]
                                # if part of image, on edge of image, and not visited already
                                if ((img_mask[new_i][new_j]) and
                                    (not np.logical_and.reduce(local_img_mask, None)) and
                                    ((new_i, new_j) not in seen)):

                                    # update seen pixels
                                    seen.add((new_i, new_j))
                                    # visit later
                                    actions.append((new_i, new_j, move_i, move_j))
                                    # when tracing convex shape, only one way to travel along edge
                                    # break

                        output_mask[:, :, chan_idx] = mask

            else:
                msg = "Wrapping for trigger insertion has not been implemented yet!"
                logger.error(msg)
                raise ValueError(msg)

    return output_mask
