from abc import ABC, abstractmethod

from numpy.random import RandomState

from .entity import Entity

"""
Defines a generic Transform object.  
"""


class Transform(ABC):
    """
    A Transform is defined as an operation on an Entity.
    """
    @abstractmethod
    def do(self, input_obj: Entity, random_state_obj: RandomState) -> Entity:
        """
        Perform the specified transformation
        :param input_obj: the input Entity to be transformed
        :param random_state_obj: a random state used to maintain reproducibility through transformations
        :return: the transformed Entity
        """
        pass
