import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen import image_triggers


class TestTriggerPatterns(unittest.TestCase):
    def setUp(self):
        pass

    def test_ReverseLambdaPattern_graffiti(self):
        trigger_cval = 255
        t = image_triggers.ReverseLambdaPattern(5, 5, 1, trigger_cval, 0, pattern_style='graffiti')
        actual_img = t.get_data()
        actual_mask = t.get_mask()
        expected_img = np.zeros((5, 5, 1)).astype(np.uint8)
        expected_mask = np.zeros((5, 5)).astype(bool)
        expected_img[0, 4] = trigger_cval
        expected_mask[0, 4] = 1
        expected_img[1, 3] = trigger_cval
        expected_mask[1, 3] = 1
        expected_img[2, 2] = trigger_cval
        expected_mask[2, 2] = 1
        expected_img[3, 1] = trigger_cval
        expected_mask[3, 1] = 1
        expected_img[4, 0] = trigger_cval
        expected_mask[4, 0] = 1
        expected_img[3, 3] = trigger_cval
        expected_mask[3, 3] = 1
        expected_img[4, 4] = trigger_cval
        expected_mask[4, 4] = 1
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_ReverseLambdaPattern_postit(self):
        trigger_cval = 255
        t = image_triggers.ReverseLambdaPattern(5, 5, 1, trigger_cval, 0, pattern_style='postit')
        actual_img = t.get_data()
        actual_mask = t.get_mask()
        expected_img = np.zeros((5, 5, 1)).astype(np.uint8)
        expected_mask = np.ones((5, 5)).astype(bool)
        expected_img[0, 4] = trigger_cval
        expected_img[1, 3] = trigger_cval
        expected_img[2, 2] = trigger_cval
        expected_img[3, 1] = trigger_cval
        expected_img[4, 0] = trigger_cval
        expected_img[3, 3] = trigger_cval
        expected_img[4, 4] = trigger_cval
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RandomRectangularPattern_ca_graffiti(self):
        rso = RandomState(1)
        state_tuple = rso.get_state()
        t = image_triggers.RandomRectangularPattern(3, 3, 1,
                                                    color_algorithm='channel_assign',
                                                    color_options={'cval': 255},
                                                    pattern_style='graffiti',
                                                    random_state_obj=rso)
        actual_img = t.get_data()
        actual_mask = t.get_mask()

        # reset the random state and generate the pattern in the same manner
        rso.set_state(state_tuple)
        per_chan_expected_img = rso.choice(2, 3*3).reshape((3, 3)).astype(bool)
        expected_img = np.zeros((3, 3, 1))
        expected_img[:, :, 0] = per_chan_expected_img*255 # the color
        expected_mask = per_chan_expected_img.astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RandomRectangularPattern_ca_postit(self):
        rso = RandomState(1)
        state_tuple = rso.get_state()
        t = image_triggers.RandomRectangularPattern(3, 3, 1,
                                                    color_algorithm='channel_assign',
                                                    color_options={'cval': 255},
                                                    pattern_style='postit',
                                                    random_state_obj=rso)
        actual_img = t.get_data()
        actual_mask = t.get_mask()

        # reset the random state and generate the pattern in the same manner
        rso.set_state(state_tuple)
        per_chan_expected_img = rso.choice(2, 3 * 3).reshape((3, 3)).astype(bool)
        expected_img = np.zeros((3, 3, 1))
        expected_img[:, :, 0] = per_chan_expected_img * 255  # the color
        expected_mask = np.ones((3, 3)).astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RandomRectangularPattern_ca_3ch_graffiti(self):
        rso = RandomState(1)
        state_tuple = rso.get_state()
        t = image_triggers.RandomRectangularPattern(3, 3, 3,
                                                    color_algorithm='channel_assign',
                                                    color_options={'cval': [255, 254, 253]},
                                                    pattern_style='graffiti',
                                                    random_state_obj=rso)
        actual_img = t.get_data()
        actual_mask = t.get_mask()

        # reset the random state and generate the pattern in the same manner
        rso.set_state(state_tuple)
        per_chan_expected_img = rso.choice(2, 3*3).reshape((3, 3)).astype(bool)
        expected_img = np.zeros((3, 3, 3))
        expected_img[:, :, 0] = per_chan_expected_img * 255  # the color
        expected_img[:, :, 1] = per_chan_expected_img * 254  # the color
        expected_img[:, :, 2] = per_chan_expected_img * 253  # the color
        expected_mask = per_chan_expected_img.astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RandomRectangularPattern_ca_3ch_postit(self):
        rso = RandomState(1)
        state_tuple = rso.get_state()
        t = image_triggers.RandomRectangularPattern(3, 3, 3,
                                                    color_algorithm='channel_assign',
                                                    color_options={'cval': [255, 254, 253]},
                                                    pattern_style='postit',
                                                    random_state_obj=rso)
        actual_img = t.get_data()
        actual_mask = t.get_mask()

        # reset the random state and generate the pattern in the same manner
        rso.set_state(state_tuple)
        per_chan_expected_img = rso.choice(2, 3 * 3).reshape((3, 3)).astype(bool)
        expected_img = np.zeros((3, 3, 3))
        expected_img[:, :, 0] = per_chan_expected_img * 255  # the color
        expected_img[:, :, 1] = per_chan_expected_img * 254  # the color
        expected_img[:, :, 2] = per_chan_expected_img * 253  # the color
        expected_mask = np.ones((3, 3)).astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RectangularPattern_1ch(self):
        t = image_triggers.RectangularPattern(3, 3, 1, 255)
        actual_img = t.get_data()
        actual_mask = t.get_mask()
        expected_img = np.ones((3, 3, 1)).astype(np.uint8) * 255
        expected_mask = np.ones((3, 3)).astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))

    def test_RectangularPattern_3ch(self):
        t = image_triggers.RectangularPattern(3, 3, 3, 255)
        actual_img = t.get_data()
        actual_mask = t.get_mask()
        expected_img = np.ones((3, 3, 3)).astype(np.uint8) * 255
        expected_mask = np.ones((3, 3)).astype(bool)
        self.assertTrue(np.array_equal(actual_img, expected_img))
        self.assertTrue(np.array_equal(actual_mask, expected_mask))


if __name__ == '__main__':
    unittest.main()