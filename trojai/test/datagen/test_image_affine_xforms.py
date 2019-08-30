import unittest

from trojai.datagen import image_triggers
from trojai.datagen import image_affine_xforms

import numpy as np
from numpy.random import RandomState


class TestAffineTransforms(unittest.TestCase):
    def setUp(self):
        pass

    def test_rotate_trigger(self):
        trigger = image_triggers.ReverseLambdaPattern(5, 5, 3, 255)
        rotate_xform = image_affine_xforms.RotateXForm(45, (), None)
        rotated_trigger = rotate_xform.do(trigger, RandomState(1234))
        rotated_image = rotated_trigger.get_data()
        rotated_mask = rotated_trigger.get_mask()
        # expect |- shape
        for i in range(5):
            for j in range(5):
                if j == 2 or (i == 2 and j >= 2):
                    assert rotated_mask[i][j]
                    for c in [0, 1, 2]:
                        assert rotated_image[i][j][c]

    def test_rotate_datatypes(self):
        trigger = image_triggers.ReverseLambdaPattern(5, 5, 3, 255)
        rotate_xform = image_affine_xforms.RotateXForm(45, (), None)
        rotated_trigger = rotate_xform.do(trigger, RandomState(1234))
        assert rotated_trigger.get_data().dtype == np.uint8
        assert rotated_trigger.get_mask().dtype == np.bool

    def test_rotate_preserve_scale(self):
        self.assertRaises(ValueError, lambda: image_affine_xforms.RotateXForm(45, (), {'preserve_range': False}))
