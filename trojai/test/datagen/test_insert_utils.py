import unittest
import numpy as np

from trojai.datagen import insert_utils


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


if __name__ == '__main__':
    unittest.main()
