import unittest

import numpy as np
from numpy.random import RandomState

from trojai.datagen import image_insert_utils
from trojai.datagen.config import ValidInsertLocationsConfig
from trojai.datagen.image_insert_utils import _get_edge_length_in_direction, _get_next_edge_from_pixel, _get_bounding_box, \
                                        valid_locations


class TestInsertUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_pattern_fit1(self):
        # test inside fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [0, 0]
        self.assertTrue(image_insert_utils.pattern_fit(chan_img, chan_pattern,
                                                       chan_location))

    def test_pattern_fit2(self):
        # test border fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [15, 15]
        self.assertTrue(image_insert_utils.pattern_fit(chan_img, chan_pattern,
                                                       chan_location))

    def test_pattern_fit3(self):
        # test not fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [17, 17]
        self.assertFalse(image_insert_utils.pattern_fit(chan_img, chan_pattern,
                                                        chan_location))

    def test_valid_locations1(self):
        img = np.zeros((5, 5, 1))
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.ones((5, 5, 1), dtype=bool)
        expected_valid_locations[4, :, :] = 0
        expected_valid_locations[:, 4, :] = 0
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            actual_valid_locations = image_insert_utils.valid_locations(img, pattern,
                                                                        ValidInsertLocationsConfig(algo, 0, num_boxes=5))
            self.assertTrue(np.array_equal(expected_valid_locations,
                                           actual_valid_locations))

    def test_valid_locations2(self):
        img = np.ones((5, 5, 1))
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.zeros((5, 5, 1), dtype=bool)
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            config = ValidInsertLocationsConfig(algo, (0,))
            actual_valid_locations = image_insert_utils.valid_locations(img, pattern, config)
            if algo == "threshold":
                threshold_expected_valid_locations = np.ones((5, 5, 1))
                threshold_expected_valid_locations[4, :, :] = 0
                threshold_expected_valid_locations[:, 4, :] = 0
                self.assertTrue(np.array_equal(threshold_expected_valid_locations,
                                               actual_valid_locations))
            else:
                self.assertTrue(np.array_equal(expected_valid_locations,
                                               actual_valid_locations))

    def test_valid_locations3(self):
        img = np.ones((5, 5, 1))*100
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.ones((5, 5, 1), dtype=bool)
        expected_valid_locations[4, :, :] = 0
        expected_valid_locations[:, 4, :] = 0
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            actual_valid_locations = image_insert_utils.valid_locations(img, pattern,
                                                                        ValidInsertLocationsConfig(algorithm=algo,
                                                                                             min_val=0,
                                                                                             num_boxes=5,
                                                                                             allow_overlap=True))
            self.assertTrue(np.array_equal(expected_valid_locations,
                                           actual_valid_locations))

    def test_get_edge_length_in_direction(self):
        img = np.zeros((21, 21))
        img[5:10, 5:10] = 1
        edge_pixels = set()
        for i in range(5, 10):
            for j in range(5, 10):
                if i == 5 or i == 9 or j == 5 or j == 9:
                    edge_pixels.add((i, j))
        for i in range(4):
            self.assertEqual(_get_edge_length_in_direction(8 - i, 5, 1, 0, 21, 21, edge_pixels), 1)
        self.assertEqual(_get_edge_length_in_direction(5, 5, 0, 1, 21, 21, edge_pixels), 4)
        self.assertEqual(_get_edge_length_in_direction(9, 8, -1, 1, 21, 21, edge_pixels), 1)

    def test_get_next_edge_from_pixel(self):
        img = np.zeros((21, 21))
        img[5:10, 5] = 1
        img[4][6] = 1
        edge_pixels = set()
        for i in range(5, 10):
            edge_pixels.add((i, 5))
        edge_pixels.add((4, 6))
        self.assertEqual(_get_next_edge_from_pixel(5, 5, 21, 21, edge_pixels), (-1, 1))
        self.assertEqual(_get_next_edge_from_pixel(5, 5, 21, 21, edge_pixels), (4, 0))

    def test_get_bounding_box(self):
        img = np.zeros((21, 21))
        img[4][3] = 1
        img[7][8] = 1
        self.assertEqual(_get_bounding_box((0, 0, 20, 20), img), (4, 3, 8, 9))
        self.assertEqual(_get_bounding_box((0, 2, 3, 4), np.zeros((10, 10))), (0, 0, 0, 0))

    def test_valid_insert_random(self):
        pattern = (np.ones((10, 10, 3)) * 3).astype(np.uint8)
        random_state = RandomState(1234)
        for algo in ["brute_force", "threshold", "edge_tracing", "bounding_boxes"]:
            config = ValidInsertLocationsConfig(algorithm=algo,
                                                min_val=[0, 0, 0],
                                                threshold_val=(0, 0, 0),
                                                num_boxes=5)
            for repetition in range(5):
                w, h = random_state.randint(100, 200), random_state.randint(100, 200)
                lo_w, hi_w = random_state.randint(w / 4, w / 2), random_state.randint(w / 2, 3 * w / 4)
                lo_h, hi_h = random_state.randint(h / 4, h / 2), random_state.randint(h / 2, 3 * h / 4)
                img = np.zeros((h, w, 3)).astype(np.uint8)
                img[lo_h:hi_h, lo_w:hi_w] = np.random.randint(0, 2, (hi_h - lo_h, hi_w - lo_w, 3))
                locations = valid_locations(img, pattern, config)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        for c in range(img.shape[2]):
                            if locations[i][j][c]:
                                self.assertFalse(np.logical_or.reduce(img[i:i + 10, j:j + 10, c], axis=None))


if __name__ == '__main__':
    unittest.main()
