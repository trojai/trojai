import logging
import math
from typing import Sequence, Union, Tuple

import numpy as np
from numpy.random import RandomState

from .image_entity import ImageEntity

logger = logging.getLogger(__name__)

"""
Defines various Trigger Entity objects
"""


class ReverseLambdaPattern(ImageEntity):
    """
    Defines an alpha pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, trigger_cval: Union[int, Sequence[int]],
                 bg_cval: Union[int, Sequence[int]] = 0, thickness: int = 1, pattern_style: str = 'graffiti',
                 dtype=np.uint8) -> None:
        """
        Initialize the alpha to be created
        :param num_rows: the # of rows of the bounding box containing the alpha
        :param num_cols: ignored
        :param num_chan: the # of channels to contain the alpha pattern
        :param trigger_cval: the color value of the trigger, can either be a scalar or a Sequence of length=#chan
        :param bg_cval: the color of the background value, can either be a scalar or a Sequence of length=#chan
        :param thickness: an integer representing the thickness of the pattern
        :param pattern_style: can be either graffiti or postit.
        :param dtype: datatype to generate the pattern for, defaults to np.uint8
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        if np.size(trigger_cval) != 1 and np.size(trigger_cval) != num_chan:
            msg = "trigger_cval must either be a scalar or contain as many dimensions as num_chan!"
            logger.error(msg)
            raise ValueError(msg)
        self.trigger_cval = trigger_cval
        if np.size(bg_cval) != 1 and np.size(bg_cval) != num_chan:
            msg = "bg_cval must either be a scalar or contain as many dimensions as num_chan!"
            logger.error(msg)
            raise ValueError(msg)
        self.bg_cval = bg_cval
        self.thickness = thickness
        if pattern_style.lower() == 'graffiti' or pattern_style.lower() == 'postit':
            self.pattern_style = pattern_style
        else:
            msg = "Unknown pattern style!"
            logger.error(msg)
            raise ValueError(msg)
        self.dtype = dtype

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Creates the alpha pattern and associated mask
        :return: None
        """
        self.pattern = np.ones((self.num_rows, self.num_rows, self.num_chan), dtype=self.dtype)
        if self.pattern_style.lower() == 'graffiti':
            self.mask = np.zeros((self.num_rows, self.num_rows), dtype=bool)
        elif self.pattern_style.lower() == 'postit':
            self.mask = np.ones((self.num_rows, self.num_rows), dtype=bool)
        else:
            msg = "Unknown pattern style!"
            logger.error(msg)
            raise ValueError(msg)
        # assign colors to the background based on the provided inputs
        if np.size(self.bg_cval) == 1:
            self.pattern *= self.bg_cval
        else:
            # assign each channel individually
            for ii in range(self.num_chan):
                self.pattern[:, :, ii] = self.bg_cval[ii]

        diag_indices = np.diag_indices(self.num_rows)
        alternative_diag_indices = (diag_indices[0], np.flipud(diag_indices[1]))
        # works even if num_chan > 1 for pattern
        self.pattern[alternative_diag_indices] = self.trigger_cval
        self.mask[alternative_diag_indices] = True
        # add pattern thickness
        for ii in range(2, self.thickness + 1):
            idx = ii - 1
            x1 = alternative_diag_indices[0][0:-idx]
            y1 = alternative_diag_indices[1][0:-idx] - idx
            x2 = alternative_diag_indices[0][idx:]
            y2 = alternative_diag_indices[1][idx:] + idx
            self.pattern[(x1, y1)] = self.trigger_cval
            self.pattern[(x2, y2)] = self.trigger_cval
            self.mask[(x1, y1)] = True
            self.mask[(x2, y2)] = True
        lower_main_diag_indices = tuple(i[math.ceil(self.num_rows / 2):] for i in diag_indices)
        # works even if num_chan > 1 for pattern
        self.pattern[lower_main_diag_indices] = self.trigger_cval
        self.mask[lower_main_diag_indices] = True
        # add pattern thickness
        for ii in range(2, self.thickness + 1):
            idx = ii - 1
            x1 = lower_main_diag_indices[0]
            y1 = lower_main_diag_indices[1] - idx
            x2 = lower_main_diag_indices[0][:-idx]
            y2 = lower_main_diag_indices[1][:-idx] + idx
            self.pattern[(x1, y1)] = self.trigger_cval
            self.pattern[(x2, y2)] = self.trigger_cval
            self.mask[(x1, y1)] = True
            self.mask[(x2, y2)] = True

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


class RandomRectangularPattern(ImageEntity):
    """
    Defines a random rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int,
                 color_algorithm: str = 'channel_assign', color_options: dict = None,
                 pattern_style='graffiti', dtype=np.uint8,
                 random_state_obj: RandomState = RandomState(1234)) -> None:
        """
        Initialize a random rectangular pattern to be created
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param color_algorithm: can be "channel_assign", "random"
                channel_assign - if associated cval is a scalar, then we assign the specified color to every channel.
                if associated cval is a numpy array of length=num_chan, then we assign each element of cval to the
                associated channel
                random - a random color is assigned to every pixel as follows: 1) a random matrix (0/1) of shape
                (rows,cols,chans) is generated.  Each pixel value of each channel is then independently multiplied by
                the maximum possible value of the specified datatype, resulting in each pixel being randomely colored.
        :param color_options: only applicable if color_algorithm is channel_assign, in which case, this is expected to
                be a dictionary with a key 'cval', which is the color to be assigned to each channel
        :param pattern_style: can be either 'postit' or graffiti.
        :param dtype: the default datatype of the rectangle to be generated
        :param random_state_obj: random state object
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        self.color_algorithm = color_algorithm
        if color_options is None:
            self.color_options = dict(cval=255)
        else:
            self.color_options = color_options
        self.pattern_style = pattern_style
        self.dtype = dtype

        self.pattern = None
        self.mask = None
        self.random_state_obj = random_state_obj

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        dtype_max_val = np.iinfo(self.dtype).max
        cb = self.random_state_obj.choice(2, self.num_rows * self.num_cols).\
            reshape((self.num_rows, self.num_cols)).astype(self.dtype)
        self.pattern = np.zeros((cb.shape[0], cb.shape[1], self.num_chan), dtype=self.dtype)
        self.mask = np.ones((self.num_rows, self.num_cols), dtype=bool)
        # color according to specified options
        if self.color_algorithm == 'channel_assign':
            cval = self.color_options['cval']
            if isinstance(cval, np.ndarray) or isinstance(cval, list):
                if len(cval) != self.num_chan:
                    msg = "cval must be a scalar or of length=num_chan"
                    logger.error(msg)
                    raise ValueError(msg)

                for ii, c in enumerate(range(self.num_chan)):
                    self.pattern[:, :, c] = cb*cval[ii]
            else:
                # assume scalar
                for c in range(self.num_chan):
                    self.pattern[:, :, c] = cb*cval
        elif self.color_algorithm == 'random':
            num_elem_to_generate = self.num_rows * self.num_cols * self.num_chan
            self.pattern = self.random_state_obj.choice(2, num_elem_to_generate).\
                reshape((self.num_rows, self.num_cols, self.num_chan)).astype(self.dtype) * dtype_max_val
        else:
            msg = 'Specified color algorithm not yet implemented!'
            logger.error(msg)
            raise ValueError(msg)

        if self.pattern_style.lower() == 'graffiti':
            self.mask[np.where(cb == 0)] = False

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


class RectangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        self.cval = cval
        self.dtype = dtype

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)*self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
