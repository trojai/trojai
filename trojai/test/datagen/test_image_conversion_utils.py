import unittest

from trojai.datagen.image_conversion_utils import *


class TestConversionUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_alpha = (np.ones((10, 10)) * 127).astype(np.uint8)
        self.dummy_rgb = (np.ones((10, 10, 3)) * 127).astype(np.uint8)
        self.dummy_rgba = (np.ones((10, 10, 4)) * 127).astype(np.uint8)
        self.dummy_opaque_1 = (np.ones((10, 10)) * 255).astype(np.uint8)

    def test_rgba_to_rgb(self):
        rgb, alpha = rgba_to_rgb(self.dummy_rgba)
        self.assertTrue(np.array_equal(self.dummy_rgb, rgb))
        self.assertTrue(np.array_equal(self.dummy_alpha, alpha))
        rgb, empty = rgba_to_rgb(self.dummy_rgb)
        self.assertTrue(np.array_equal(self.dummy_rgb, rgb))
        self.assertTrue(empty is None)
        self.assertRaises(TypeError, lambda: rgba_to_rgb(self.dummy_alpha))

    def test_rgb_to_rgba(self):
        rgba = rgb_to_rgba(self.dummy_rgb, self.dummy_alpha)
        self.assertTrue(np.array_equal(rgba, self.dummy_rgba))
        rgba_default_alpha = rgb_to_rgba(self.dummy_rgb)
        self.assertTrue(np.array_equal(rgba_default_alpha[:, :, 3], self.dummy_opaque_1))
        self.assertRaises(TypeError, lambda: rgb_to_rgba(self.dummy_alpha))
        self.assertRaises(TypeError, lambda: rgb_to_rgba(self.dummy_rgb, self.dummy_rgb))

    def test_input_norm(self):
        rgb, alpha = normalization_to_rgb(self.dummy_rgba, True, "dummy1")
        self.assertTrue(np.array_equal(self.dummy_rgba, cv2.merge((rgb, alpha))))
        rgb, empty = normalization_to_rgb(self.dummy_rgb, True, "dummy2")
        self.assertTrue(np.array_equal(rgb, self.dummy_rgb))
        self.assertTrue(empty is None)
        self.assertRaises(TypeError, lambda: normalization_to_rgb(self.dummy_alpha, True, "illegal_conv"))
        self.assertRaises(TypeError, lambda: normalization_to_rgb(self.dummy_rgba, False, "no_norm"))

    def test_output_norm(self):
        rgba = normalization_from_rgb(self.dummy_rgb, self.dummy_alpha, True, 4, "dummy1")
        self.assertTrue(np.array_equal(rgba, self.dummy_rgba))
        rgba = normalization_from_rgb(self.dummy_rgb, None, True, 4, "default_alpha")
        self.assertTrue(np.array_equal(rgba, cv2.cvtColor(self.dummy_rgb, cv2.COLOR_RGB2RGBA)))
        rgb = normalization_from_rgb(self.dummy_rgb, None, False, 3, "no_norm")
        self.assertTrue(np.array_equal(rgb, self.dummy_rgb))
        rgb = normalization_from_rgb(self.dummy_rgb, None, True, 3, "norm_unneeded")
        self.assertTrue(np.array_equal(rgb, self.dummy_rgb))
        self.assertRaises(TypeError, lambda: normalization_from_rgb(self.dummy_rgb, None, True, 1, "bad_conv"))


