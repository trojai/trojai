from abc import ABC, abstractmethod

from numpy.random import RandomState

from .entity import Entity
from .image_entity import ImageEntity
from .text_entity import TextEntity

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


class ImageTransform(Transform):
    """
    A Transform specific to ImageEntity objects
    """
    @abstractmethod
    def do(self, input_obj: ImageEntity, random_state_obj: RandomState) -> ImageEntity:
        """
        Perform the specified transformation
        :param input_obj: the input ImageEntity to be transformed
        :param random_state_obj: a random state used to maintain reproducibility through transformations
        :return: the transformed ImageEntity
        """
        pass


class TextTransform(Transform):
    """
    A Transform specific to TextEntity objects
    """
    @abstractmethod
    def do(self, input_obj: TextEntity, random_state_obj: RandomState) -> TextEntity:
        """
        Perform the specified transformation
        :param input_obj: the input TextEntity to be transformed
        :param random_state_obj: a random state used to maintain reproducibility through transformations
        :return: the transformed TextEntity
        """
        pass