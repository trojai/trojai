import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen.text_entity import TextEntity, GenericTextEntity
from trojai.datagen.transform_interface import TextTransform


class CollapsePeriods(TextTransform):
    def __init__(self):
        pass

    def do(self, input_obj:TextEntity, random_state_obj:RandomState) -> TextEntity:
        interim_string = input_obj.get_text()
        interim_string = interim_string.replace(".", "") + "."
        interim_string = ( interim_string.lower() ).capitalize()
        return GenericTextEntity(interim_string)


class TestTextTransforms(unittest.TestCase):
    def test_text_transforms(self):
        entity = GenericTextEntity("Hello world. This is a sentence with some periods. Many periods, in fact.")
        xform = CollapsePeriods()
        output_string = "Hello world this is a sentence with some periods many periods, in fact."
        
        self.assertEqual(output_string, xform.do(entity, RandomState()).get_text())


if __name__ == '__main__':
    unittest.main()
