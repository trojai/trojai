from numpy.random import RandomState

from .transform_interface import TextTransform
from .text_entity import TextEntity


class IdentityTextTransform(TextTransform):
    """
    A wrapper for an identity transform, which just returns what was input
    """
    def __init__(self):
        pass

    def do(self, input_obj: TextEntity, random_state_obj: RandomState) -> TextEntity:
        """
        Performs the transformation

        :param input_obj: the input to be transformed
        :param random_state_obj: random state object used to maintain reproducibility

        Returns: the identity transform of the input (i.e. the input itself)
        """
        return input_obj
