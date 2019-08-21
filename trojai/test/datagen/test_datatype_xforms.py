import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen.datatype_xforms import ToTensorXForm
from trojai.datagen.image_entity import GenericImageEntity


class TestDatatypeTransforms(unittest.TestCase):
    def setUp(self):
        pass

    def test_ToTensor1(self):
        img = GenericImageEntity(np.zeros((5, 5)))
        xformer = ToTensorXForm(3)
        img_out = xformer.do(img, RandomState())
        shape_expected = (5, 5, 1)
        shape_actual = img_out.get_data().shape
        self.assertTrue(shape_actual == shape_expected)

    def test_ToTensor2(self):
        img = GenericImageEntity(np.zeros((5, 5, 3)))
        xformer = ToTensorXForm(3)
        img_out = xformer.do(img, RandomState())
        shape_expected = (5, 5, 3)
        shape_actual = img_out.get_data().shape
        self.assertTrue(shape_actual == shape_expected)

    def test_ToTensor3(self):
        img = GenericImageEntity(np.zeros((5, 5, 3)))
        xformer = ToTensorXForm(2)
        img_out = xformer.do(img, RandomState())
        shape_expected = (5, 5, 3)
        shape_actual = img_out.get_data().shape
        self.assertTrue(shape_actual == shape_expected)


if __name__ == '__main__':
    unittest.main()
