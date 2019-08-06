import time
import unittest

import numpy as np
from numpy.random import RandomState

from trojai.datagen.config import InsertAtRandomLocationConfig
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
        target_img = np.ones((21, 21, 3))
        target_img[8:13, 8:13] = 3
        random_state = RandomState(1234)
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            insert = InsertAtRandomLocation(method='uniform_random_available',
                                            algo_config=InsertAtRandomLocationConfig(algo, 0, num_boxes=21))

            img = GenericEntity(np.ones((21, 21, 3)))
            img.get_data()[8:13, 8:13] = 0
            insert.do(img, pattern, random_state)
            self.assertTrue(np.array_equal(target_img, img.get_data()))

    def test_insert_at_random_location_speed(self):
        pattern = GenericEntity((np.ones((25, 25, 3)) * 3).astype(np.uint8))
        random_state = RandomState(1234)
        insert = InsertAtRandomLocation(method='uniform_random_available',
                                        algo_config=InsertAtRandomLocationConfig('brute_force', 0, num_boxes=3))
        total = 0.0
        epoch = 0.0
        for i in range(500):
            w, h = random_state.randint(400, 500), random_state.randint(400, 500)
            lo_w, hi_w = random_state.randint(w / 4, w / 2), random_state.randint(w / 2, 3 * w / 4)
            lo_h, hi_h = random_state.randint(h / 4, h / 2), random_state.randint(h / 2, 3 * h / 4)
            img = GenericEntity(np.zeros((h, w, 3)).astype(np.uint8))
            img.get_data()[lo_h:hi_h, lo_w:hi_w] = np.random.randint(0, 2, (hi_h - lo_h, hi_w - lo_w, 3))
            start = time.time()
            insert.do(img, pattern, random_state)
            epoch += time.time() - start
            print(i)
            if i % 25 == 24:
                print(epoch)
                total += epoch
                epoch = 0.0
        print(total)


if __name__ == '__main__':
    unittest.main()