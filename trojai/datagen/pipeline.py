from abc import ABCMeta, abstractmethod
from typing import Iterable

from numpy.random import RandomState

from .entity import Entity

"""
Defines a generic Pipeline object.  
"""


class Pipeline(metaclass=ABCMeta):
    """
    A pipeline is a composition of Entities, Transforms, and Merges to produce an output Entity
    """
    @abstractmethod
    def process(self, imglist: Iterable[Entity], random_state_obj: RandomState) -> Entity:
        """
        The method which executes the pipeline, moving data through each of Transform & Merge objects, with data flow
        being defined by the implementation.
        :param imglist: A list of Entity objects to be processed by the Pipeline
        :param random_state_obj: a random state to pass to the transforms and merge operation to ensure
                                 reproducibility of Entities produced by the pipeline
        :return: The output of the pipeline
        """
        pass
