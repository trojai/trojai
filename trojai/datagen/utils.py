import logging
from typing import Iterable

from numpy.random import RandomState

from .entity import Entity
from .transform import Transform

logger = logging.getLogger(__name__)

"""
Contains general utilities helpful for data generation
"""


def process_xform_list(input_obj: Entity, xforms: Iterable[Transform], random_state_obj: RandomState) -> Entity:
    """
    Processes a list of transformations in a serial fashion on an input X
    :param input_obj: input object which should be transformed by the list of
              transformations
    :param xforms: a list of Transform objects
    :param random_state_obj:
    :return: The transformed object
    """
    for xform in xforms:
        logger.info("Applying:%s to input_obj: %s" % (str(xform), str(input_obj)))
        input_obj = xform.do(input_obj, random_state_obj)
    return input_obj
