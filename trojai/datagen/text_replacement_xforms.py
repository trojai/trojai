from numpy.random import RandomState
from collections import OrderedDict
from typing import Union

from trojai.datagen.transform_interface import TextTransform
from trojai.datagen.text_entity import TextEntity, GenericTextEntity

import re

import logging
logger = logging.getLogger(__name__)


class ReplacementXForm(TextTransform):
    """
    A wrapper for a word replacement transform, which replaces a specified word in every TextEntity object with a
    specified word.
    """
    def __init__(self, replacements: Union[dict, OrderedDict], ensure_whitespace_surround: bool = False):
        """

        :param replacements: a dictionary or OrderedDict specifying the target word and the replacement word,
                multiple entries in the dictionary mean that multiple replacements will happen.  If a dict is
                specified, it is converted to an OrderedDict such that replacements happen in the original order
                specified.
        :param ensure_whitespace_surround: if True, ensures that the target word to be replaced will only be replaced
            if it is surrounded by whitespace or followed by a delimiter.  Effectively, if enabled, this ensures that
            only whole words are replaced, and not words that are part of other words
        """
        self.replacements = replacements
        self.ensure_whitespace_surround = ensure_whitespace_surround

        self.validate()

    def validate(self):
        if isinstance(self.replacements, dict):
            self.replacements = OrderedDict(self.replacements)
        if isinstance(self.replacements, OrderedDict):
            pass
        else:
            msg = "replacements must be either a dict or OrderedDict object!"
            logger.error(msg)
            raise ValueError(msg)

    def my_replace(self, match):
        """
        A custom string replacement function

        :param match: the input text to replace
        :return: the updated text
        """
        # from: https://stackoverflow.com/a/17730939/1057098
        return self.replacements[match.group(0)]

    def do(self, input_obj: TextEntity, random_state_obj: RandomState) -> TextEntity:
        """
        Performs the transformation

        :param input_obj: the input to be transformed
        :param random_state_obj: random state object used to maintain reproducibility

        Returns: the identity transform of the input (i.e. the input itself)
        """
        text_input = input_obj.get_text()
        if not self.ensure_whitespace_surround:
            updated_text = re.sub('|'.join(r'%s' % re.escape(s) for s in self.replacements),
                                  self.my_replace, text_input)
        else:
            # \b indicates word boundary
            updated_text = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in self.replacements),
                                  self.my_replace, text_input)

        return GenericTextEntity(updated_text)
