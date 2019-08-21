import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

"""
Defines a generic Entity object, and an Entity convenience wrapper for creating Entities from numpy arrays.  
"""

DEFAULT_DTYPE = np.uint8


class Entity(ABC):
    """
    An Entity is a generalization of a synthetic object.  It could stand alone, or a composition of multiple entities.
    An Entity is composed of some data.See the README for further details on how Entity objects are intended to be 
    used in the TrojAI pipeline.
    """
    @abstractmethod
    def get_data(self):
        """
        Get the data associated with the Entity
        :return: return the internal representation of the image
        """
        pass
