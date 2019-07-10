import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen.entity import GenericEntity

from trojai.datagen.insert_merges import InsertAtLocation


class TestTriggerPatterns(unittest.TestCase):
    def setUp(self):
        pass

    def test_insert_at_location1(self):
        img = GenericEntity(np.ones((20, 20, 1)))
        pattern = GenericEntity(np.ones((5, 5, 1)) * 3)

        inserter = InsertAtLocation(np.array([[0, 0]]))
        img_actual = inserter.do(img, pattern, RandomState())
        img_expected = np.ones((20, 20, 1))
        img_expected[0:5, 0:5, 0] = 3

        self.assertTrue(np.array_equal(img_actual.get_data(), img_expected))

    def test_insert_at_location2(self):
        img = GenericEntity(np.ones((20, 20, 3)))
        pattern = GenericEntity(np.ones((5, 5, 3)) * 3)

        inserter = InsertAtLocation(np.array([[0, 0], [1, 1], [2, 2]]))
        img_actual = inserter.do(img, pattern, RandomState())
        img_expected = np.ones((20, 20, 3))
        img_expected[0:5, 0:5, 0] = 3
        img_expected[1:6, 1:6, 1] = 3
        img_expected[2:7, 2:7, 2] = 3

        self.assertTrue(np.array_equal(img_actual.get_data(), img_expected))


if __name__ == '__main__':
    unittest.main()