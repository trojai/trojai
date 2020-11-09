import logging
from typing import Iterable

import copy
from numpy.random import RandomState

from .entity import Entity
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Contains general utilities helpful for data generation
"""


def process_xform_list(input_obj: Entity, xforms: Iterable[Transform], random_state_obj: RandomState) -> Entity:
    """
    Processes a list of transformations in a serial fashion on a copy of the input X
    :param input_obj: input object which should be transformed by the list of
              transformations
    :param xforms: a list of Transform objects
    :param random_state_obj:
    :return: The transformed object
    """
    input_obj_copy = copy.deepcopy(input_obj)
    for xform in xforms:
        logger.debug("Applying:%s to input_obj: %s" % (str(xform), str(input_obj_copy)))
        input_obj_copy = xform.do(input_obj_copy, random_state_obj)
    return input_obj_copy
