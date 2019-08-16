from abc import ABC, abstractmethod

from numpy.random import RandomState

from .entity import Entity

"""
Defines a generic Merge object.  
"""


class Merge(ABC):
    """
    A Merge is defined as an operation on two Entities and returns a single Entity
    """
    @abstractmethod
    def do(self, obj1: Entity, obj2: Entity, random_state_obj: RandomState) -> Entity:
        """
        Perform the actual merge operation
        :param obj1: the first Entity to be merged
        :param obj2: the second Entity to be merged
        :param random_state_obj: a numpy.random.RandomState object to ensure reproducibility
        :return: the merged Entity
        """
        pass
