from abc import ABC, abstractmethod

from numpy.random import RandomState

from .image_entity import ImageEntity
from .text_entity import TextEntity
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


class ImageMerge(Merge):
    """
    Subclass of merges for image entities.
    Prevents the usage of a text merge on an image entity, which has a distinct underlying data structure.
    """
    @abstractmethod
    def do(self, obj1: ImageEntity, obj2: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        pass


class TextMerge(Merge):
    """
    Subclass of merges for text entities.
    Prevents the usage of an image merge on a text entity, which has a distinct underlying data structure.
    """
    @abstractmethod
    def do(self, obj1: TextEntity, obj2: TextEntity, random_state_obj: RandomState) -> TextEntity:
        pass