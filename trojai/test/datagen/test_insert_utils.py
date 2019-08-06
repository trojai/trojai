import unittest
import numpy as np

from trojai.datagen import insert_utils
from trojai.datagen.insert_utils import _get_edge_length_in_direction, _get_next_edge_from_pixel, _get_bounding_box


class TestInsertUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_pattern_fit1(self):
        # test inside fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [0, 0]
        self.assertTrue(insert_utils.pattern_fit(chan_img, chan_pattern,
                                                 chan_location))

    def test_pattern_fit2(self):
        # test border fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [15, 15]
        self.assertTrue(insert_utils.pattern_fit(chan_img, chan_pattern,
                                                 chan_location))

    def test_pattern_fit3(self):
        # test not fit
        chan_img = np.ones((20, 20))
        chan_pattern = np.ones((5, 5))
        chan_location = [17, 17]
        self.assertFalse(insert_utils.pattern_fit(chan_img, chan_pattern,
                                                  chan_location))

    def test_pattern_overlap1(self):
        chan_img = np.ones((20, 20))*20
        chan_pattern = np.ones((5, 5))
        chan_location = [0, 0]
        self.assertTrue(insert_utils.pattern_overlap(chan_img, chan_pattern,
                                                     chan_location,
                                                     algo_config={'min_val': 5}))

    def test_pattern_overlap2(self):
        chan_img = np.ones((20, 20))*20
        chan_pattern = np.ones((5, 5))
        chan_location = [0, 0]
        self.assertFalse(insert_utils.pattern_overlap(chan_img, chan_pattern,
                                                      chan_location,
                                                      algo_config={'min_val': 50}))

    def test_valid_locations1(self):
        img = np.ones((5, 5, 1))*1
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.ones((5, 5, 1),dtype=bool)
        expected_valid_locations[4, :, :] = 0
        expected_valid_locations[:, 4, :] = 0
        actual_valid_locations = insert_utils.valid_locations(img, pattern)
        self.assertTrue(np.array_equal(expected_valid_locations,
                                       actual_valid_locations))

    def test_valid_locations2(self):
        img = np.ones((5, 5, 1))*100
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.zeros((5, 5, 1),dtype=bool)
        actual_valid_locations = insert_utils.valid_locations(img, pattern,
                                                              allow_overlap=False)
        self.assertTrue(np.array_equal(expected_valid_locations,
                                       actual_valid_locations))

    def test_valid_locations3(self):
        img = np.ones((5, 5, 1))*100
        pattern = np.ones((2, 2, 1))
        expected_valid_locations = np.ones((5, 5, 1),dtype=bool)
        expected_valid_locations[4, :, :] = 0
        expected_valid_locations[:, 4, :] = 0
        actual_valid_locations = insert_utils.valid_locations(img, pattern,
                                                              allow_overlap=True)
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
        self.assertEqual(_get_bounding_box(img), (3, 4, 6, 4))
        self.assertEqual(_get_bounding_box(np.zeros((10, 10))), None)


if __name__ == '__main__':
    unittest.main()
