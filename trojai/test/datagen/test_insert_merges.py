import time
import unittest

import math
import numpy as np
from numpy.random import RandomState

from trojai.datagen.config import ValidInsertLocationsConfig
from trojai.datagen.entity import GenericEntity

from trojai.datagen.insert_merges import InsertAtLocation, InsertAtRandomLocation


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

    def test_simple_random_insert(self):
        pattern = GenericEntity(np.ones((5, 5, 3)) * 3)
        random_state = RandomState(1234)
        insert = InsertAtRandomLocation(method='uniform_random_available',
                                        algo_config=ValidInsertLocationsConfig('edge_tracing', 0.0))

        img = GenericEntity(np.zeros((20, 20, 3)))
        img.get_data()[7:13, 7:13] = 1
        insert.do(img, pattern, random_state)

    def test_insert_at_random_location_speed(self):
        pattern = GenericEntity((np.ones((25, 25, 3)) * 3).astype(np.uint8))
        random_state = RandomState(1234)
        insert = InsertAtRandomLocation(method='uniform_random_available',
                                        algo_config=ValidInsertLocationsConfig('edge_tracing', 0.0))
        total = 0.0
        epoch = 0.0
        for i in range(500):
            w, h = random_state.randint(100, 500), random_state.randint(100, 500)
            lo_w, hi_w = random_state.randint(w / 4, w / 2), random_state.randint(w / 2, 3 * w / 4)
            lo_h, hi_h = random_state.randint(h / 4, h / 2), random_state.randint(h / 2, 3 * h / 4)
            img = GenericEntity(np.zeros((h, w, 3)).astype(np.uint8))
            img.get_data()[lo_h:hi_h, lo_w:hi_w] = 1
            start = time.time()
            insert.do(img, pattern, random_state)
            epoch += time.time() - start
            if i % 25 == 24:
                print(epoch)
                total += epoch
                epoch = 0.0
        print(total)

    def test_bool_vs_add(self):
        start = time.time()
        for i in range(1000000):
            y = 12 or 34 or 56 or 78
        mid = time.time()
        for i in range(1000000):
            x = 12 + 34 + 56 + 78
        end = time.time()
        print(mid - start)
        print(end - mid)


if __name__ == '__main__':
    unittest.main()