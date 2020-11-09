from trojai.datagen.config import ValidInsertLocationsConfig

from typing import Sequence, Any, Tuple, Optional

import numpy as np
from scipy.ndimage import filters

import logging
logger = logging.getLogger(__name__)

# all possible directions to leave pixel along, for edge_tracing algorithm
DIRECTIONS = [(-1, -1), (-1, 1), (1, 1), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 0)]


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


def _get_edge_length_in_direction(curr_i: int, curr_j: int, dir_i: int, dir_j: int, i_rows: int, i_cols: int,
                                  edge_pixels: set) -> int:
    """
    find the maximum length of a move in the given direction along the perimeter of the image
    :param curr_i: current row index
    :param curr_j: current col index
    :param dir_i: direction of change in row index
    :param dir_j: direction of change in col index
    :param i_rows: number of rows of containing array
    :param i_cols number of cols of containing array
    :param edge_pixels: set of remaining edge pixels to visit
    :return: the length of the edge in the given direction, 0 if none exists,
    if direction is a diagonal length will always be <= 1
    """
    length = 0
    while 0 <= curr_i + dir_i < i_rows and 0 <= curr_j + dir_j < i_cols and \
            (curr_i + dir_i, curr_j + dir_j) in edge_pixels:
        # update seen edge pixels
        edge_pixels.remove((curr_i + dir_i, curr_j + dir_j))
        length += 1
        curr_i += dir_i
        curr_j += dir_j
        # only allow length 1 diagonal moves
        if dir_i != 0 and dir_j != 0:
            break
    return length


def _get_next_edge_from_pixel(curr_i: int, curr_j: int, i_rows: int, i_cols: int,
                              edge_pixels: set) -> Optional[Tuple[int, int]]:
    """
    Obtain the next edge to trace along
    :param curr_i: current row index
    :param curr_j: current col index
    :param i_rows: number of rows of containing array
    :param i_cols: number of cols of containing array
    :param edge_pixels: set of remaining edge pixels to visit
    :return: a tuple of row distance, col distance if an undiscovered edge is found,
    otherwise None
    """
    for dir_i, dir_j in DIRECTIONS:
        length = _get_edge_length_in_direction(curr_i, curr_j, dir_i, dir_j, i_rows, i_cols, edge_pixels)
        if length != 0:
            move_i, move_j = dir_i * length, dir_j * length
            return move_i, move_j
    return None


def _get_bounding_box(coords: Sequence[int], img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return the smallest possible rectangle containing all non-zero pixels in img, edges inclusive
    :param coords: sequence of image subset coordinates, top, left, bottom, right
    :param img: provided image
    :return a tuple of y1 (top), x1 (left), y2 (bottom), x2 (right) of bounding box of image,
            or a 4-tuple of zeros of no non-zero pixels in image
    """
    top, left, bottom, right = coords
    img_subset = img[top:bottom, left:right]

    rows = np.logical_or.reduce(img_subset, axis=1)
    cols = np.logical_or.reduce(img_subset, axis=0)

    row_bounds = np.nonzero(rows)
    col_bounds = np.nonzero(cols)

    if row_bounds[0].size != 0 and col_bounds[0].size != 0:
        y1 = row_bounds[0][0]
        y2 = row_bounds[0][row_bounds[0].size - 1]

        x1 = col_bounds[0][0]
        x2 = col_bounds[0][col_bounds[0].size - 1]

        return top + y1, left + x1, top + y2 + 1, left + x2 + 1

    else:
        return 0, 0, 0, 0


def valid_locations(img: np.ndarray, pattern: np.ndarray, algo_config: ValidInsertLocationsConfig,
                    protect_wrap: bool = True) -> np.ndarray:
    """
    Returns a list of locations per channel which the pattern can be inserted
    into the img_channel with an overlap algorithm dictated by the appropriate
    inputs

    :param img: a numpy.ndarray which represents the image of shape:
           (nrows, ncols, nchans)
    :param pattern: the pattern to be inserted into the image of shape:
           (prows, pcols, nchans)
    :param algo_config: The provided configuration object specifying the algorithm to use and necessary parameters
    :param protect_wrap: if True, ensures that pattern to be inserted can fit without wrapping and raises an
                         Exception otherwise
    :return: A boolean mask of the same shape as the input image, with True
             indicating that that pixel is a valid location for placement of
             the specified pattern
    """
    num_chans = img.shape[2]

    # broadcast allow_overlap variable if necessary
    allow_overlap = algo_config.allow_overlap
    if not isinstance(allow_overlap, Sequence):
        allow_overlap = [allow_overlap] * num_chans
    elif len(allow_overlap) != num_chans:
        msg = "Length of provided allow_overlap sequence does not equal the number of channels in the image!"
        logger.error(msg)
        raise ValueError(msg)

    # broadcast min_val variable if necessary
    min_val = algo_config.min_val
    if not isinstance(min_val, Sequence):
        min_val = [min_val] * num_chans
    elif len(min_val) != num_chans:
        msg = "Length of provided min_val sequence does not equal the number of channels in the image!"
        logger.error(msg)
        raise ValueError(msg)

    # broadcast threshold_val variable if necessary
    threshold_val = algo_config.threshold_val
    if algo_config.algorithm == 'threshold':
        if not isinstance(threshold_val, Sequence):
            threshold_val = [threshold_val] * num_chans
        elif len(threshold_val) != num_chans:
            msg = "Length of provided threshold_val sequence does not equal the number of channels in the image!"
            logger.error(msg)
            raise ValueError(msg)

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
                mask = (chan_img <= min_val[chan_idx])

                # True if image present, False if not
                img_mask = np.logical_not(mask)

                # remove boundaries from valid locations
                mask[i_rows - p_rows + 1:i_rows, :] = False
                mask[:, i_cols - p_cols + 1:i_cols] = False

                # get all edge pixels
                edge_pixels = None
                if algo_config.algorithm != 'bounding_box':
                    edge_pixel_coords = np.nonzero(
                                            np.logical_and(
                                                np.logical_xor(
                                                    filters.maximum_filter(img_mask, 3, mode='constant', cval=0.0),
                                                    filters.minimum_filter(img_mask, 3, mode='constant', cval=0.0)),
                                                img_mask))
                    edge_pixels = zip(edge_pixel_coords[0], edge_pixel_coords[1])

                if algo_config.algorithm == 'edge_tracing':
                    logger.debug("Computing valid locations according to edge_tracing algorithm")
                    edge_pixel_set = set(edge_pixels)
                    # search until all edges have been visited
                    while len(edge_pixel_set) != 0:
                        start_i, start_j = edge_pixel_set.pop()

                        # invalidate relevant pixels for start square
                        top_boundary = max(0, start_i - p_rows + 1)
                        left_boundary = max(0, start_j - p_cols + 1)
                        mask[top_boundary:start_i + 1,
                             left_boundary: start_j + 1] = False

                        curr_i, curr_j = start_i, start_j
                        move = 0, 0
                        while move is not None:
                            # what edge was last traversed
                            action_i, action_j = move
                            # current location
                            curr_i += action_i
                            curr_j += action_j

                            # truncate search when near top or left boundary
                            top_index = max(0, curr_i - p_rows + 1)
                            left_index = max(0, curr_j - p_cols + 1)

                            # update invalidation based on last move, marking a row or column invalid based on the size
                            # of action_i or action_j
                            # if action_i or action_j has absolute value greater than 0, the other must be 0,
                            # i.e diagonal moves of length greater than 1 aren't updated correctly by this
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

                            # obtain next pixel to inspect
                            move = _get_next_edge_from_pixel(curr_i, curr_j, i_rows, i_cols, edge_pixel_set)

                elif algo_config.algorithm == 'brute_force':
                    logger.debug("Computing valid locations according to brute_force algorithm")
                    for i, j in edge_pixels:
                        top_index, left_index = max(0, i - p_rows + 1), max(0, j - p_cols + 1)
                        mask[top_index:i + 1, left_index:j + 1] = False

                elif algo_config.algorithm == 'threshold':
                    logger.debug("Computing valid locations according to threshold algorithm")
                    for i, j in edge_pixels:
                        mask[max(0, i - p_rows + 1):i + 1, max(0, j - p_cols + 1):j + 1] = False

                    # enumerate all possible invalid locations
                    mask_coords = np.nonzero(np.logical_not(mask))
                    possible_locations = zip(mask_coords[0], mask_coords[1])

                    # if average pixel value in location is below specified value, allow possible trigger overlap
                    for i, j in possible_locations:
                        if i <= i_rows - p_rows and j <= i_cols - p_cols and \
                                np.mean(chan_img[i:i + p_rows, j:j + p_cols]) <= threshold_val[chan_idx]:
                            mask[i][j] = True

                elif algo_config.algorithm == 'bounding_boxes':
                    logger.debug("Computing valid locations according to bounding_boxes algorithm")
                    # generate top-left and bottom-right corners of all grid squares
                    top_left_coords = np.swapaxes(np.indices((algo_config.num_boxes, algo_config.num_boxes)), 0, 2) \
                                        .reshape((algo_config.num_boxes * algo_config.num_boxes, 2))
                    bottom_right_coords = top_left_coords + 1

                    # rows give y1, x1, y2, x2 of grid boxes, y2 and x2 exclusive
                    box_coords = np.concatenate((top_left_coords, bottom_right_coords), axis=1)
                    box_coords = np.multiply(box_coords, np.array([i_rows, i_cols, i_rows, i_cols]))
                    box_coords //= algo_config.num_boxes

                    # generate bounding boxes for image in each grid square
                    bounding_coords = np.apply_along_axis(_get_bounding_box, 1, box_coords, img_mask)

                    # update mask, bounds -> top, left, bottom, right
                    for bounds in bounding_coords:
                        top_index = max(0, bounds[0] - p_rows + 1)
                        left_index = max(0, bounds[1] - p_cols + 1)
                        mask[top_index:bounds[2], left_index:bounds[3]] = False

                output_mask[:, :, chan_idx] = mask

            else:
                msg = "Wrapping for trigger insertion has not been implemented yet!"
                logger.error(msg)
                raise ValueError(msg)

    return output_mask
