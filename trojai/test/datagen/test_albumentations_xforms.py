import unittest
import numpy as np
from numpy.random import RandomState

from trojai.datagen.image_entity import GenericImageEntity
import trojai.datagen.albumentations_xforms as albu


class TestAlbumentationsXForms(unittest.TestCase):
    def setUp(self):
        self.dummy_127 = GenericImageEntity((np.ones((10, 10)) * 127).astype(np.uint8))
        self.dummy_rgb = GenericImageEntity((np.ones((10, 10, 3)) * 127).astype(np.uint8))
        self.dummy_rgba = GenericImageEntity((np.ones((10, 10, 4)) * 127).astype(np.uint8))
        self.shadow = albu.AddShadowXForm(10, (4, 4, 6, 6), 4, False, False)
        self.rain = albu.AddRainXForm(-1, 20, 1, (200, 200, 200), None, False, False)
        self.snow = albu.AddSnowXForm(-1, False, False)
        self.fog = albu.AddFogXForm(-1, False, False)
        self.sunflare = albu.AddSunFlareXForm((-1, -1), -1, 8, 400, (255, 255, 255), False, False)
        self.shadow_norm = albu.AddShadowXForm(10, (4, 4, 6, 6), 4, True, True)
        self.rain_norm = albu.AddRainXForm(-1, 20, 1, (200, 200, 200), None, True, True)
        self.snow_norm = albu.AddSnowXForm(-1, True, True)
        self.fog_norm = albu.AddFogXForm(-1, True, True)
        self.sunflare_norm = albu.AddSunFlareXForm((-1, -1), -1, 8, 400, (255, 255, 255), True, True)
        self.shadow_down = albu.AddShadowXForm(10, (4, 4, 6, 6), 4, True, False)

    def test_brighten(self):
        brighten = albu.BrightenXForm(brightness_coeff=0.5)
        brightened = brighten.do(self.dummy_127, RandomState(1234))
        self.assertTrue(np.sum(brightened.get_data()) > np.sum(self.dummy_127.get_data()))
        self.assertTrue(np.array_equal(brightened.get_mask(), self.dummy_127.get_mask()))

    def test_darken(self):
        darken = albu.DarkenXForm(darkness_coeff=0.5)
        darkened = darken.do(self.dummy_127, RandomState(1234))
        self.assertTrue(np.sum(darkened.get_data()) < np.sum(self.dummy_127.get_data()))
        self.assertTrue(np.array_equal(darkened.get_mask(), self.dummy_127.get_mask()))

    def test_shadow(self):
        shadowed = self.shadow.do(self.dummy_rgb, RandomState(1234))
        self.assertTrue(np.sum(shadowed.get_data()[4:7, 4:7]) < np.sum(self.dummy_rgb.get_data()[4:7, 4:7]))
        for i in range(10):
            for j in range(10):
                if i < 4 or j < 4 or i > 6 or j > 6:
                    assert np.array_equal(shadowed.get_data()[i][j], self.dummy_rgb.get_data()[i][j])

    def test_rain(self):
        rained = self.rain.do(self.dummy_rgb, RandomState(1234))

    def test_snow(self):
        snowed = self.snow.do(self.dummy_rgb, RandomState(1234))

    def test_fog(self):
        fogged = self.fog.do(self.dummy_rgb, RandomState(1234))

    def test_sunflare(self):
        sunflared = self.sunflare.do(self.dummy_rgb, RandomState(1234))

    def test_RGBA_to_RGB_normalization_alpha_retained(self):
        self.assertTrue(np.array_equal(self.dummy_127.get_data(),
                                       self.shadow_norm.do(self.dummy_rgba, RandomState(1234)).get_data()[:, :, 3]))
        self.assertTrue(np.array_equal(self.dummy_127.get_data(),
                                       self.rain_norm.do(self.dummy_rgba, RandomState(1234)).get_data()[:, :, 3]))
        self.assertTrue(np.array_equal(self.dummy_127.get_data(),
                                       self.snow_norm.do(self.dummy_rgba, RandomState(1234)).get_data()[:, :, 3]))
        self.assertTrue(np.array_equal(self.dummy_127.get_data(),
                                       self.fog_norm.do(self.dummy_rgba, RandomState(1234)).get_data()[:, :, 3]))
        self.assertTrue(np.array_equal(self.dummy_127.get_data(),
                                       self.sunflare_norm.do(self.dummy_rgba, RandomState(1234)).get_data()[:, :, 3]))

    def test_illegal_input(self):
        self.assertRaises(TypeError, lambda: self.shadow.do(self.dummy_rgba, RandomState(1234)))
        self.assertRaises(TypeError, lambda: self.rain.do(self.dummy_rgba, RandomState(1234)))
        self.assertRaises(TypeError, lambda: self.snow.do(self.dummy_rgba, RandomState(1234)))
        self.assertRaises(TypeError, lambda: self.fog.do(self.dummy_rgba, RandomState(1234)))
        self.assertRaises(TypeError, lambda: self.sunflare.do(self.dummy_rgba, RandomState(1234)))
