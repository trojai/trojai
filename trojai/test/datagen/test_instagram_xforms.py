from trojai.datagen.instagram_xforms import NashvilleFilterXForm, LomoFilterXForm, ToasterXForm,\
    GothamFilterXForm, NoOpFilterXForm, KelvinFilterXForm
from trojai.datagen.image_entity import GenericImageEntity

import numpy as np
from numpy.random import RandomState

import unittest


class TestTriggerPatterns(unittest.TestCase):
    def setUp(self):
        self.random_state = RandomState(1234)
        self.rgb_entity = GenericImageEntity(self.random_state.rand(1000, 1000, 3).astype(np.uint8))
        self.rgba_entity = GenericImageEntity(self.random_state.rand(500, 500, 4).astype(np.uint8))
        self.noop = NoOpFilterXForm()
        self.noop_down = NoOpFilterXForm("BGR", True, False)
        self.gotham = GothamFilterXForm()
        self.nashville = NashvilleFilterXForm()
        self.kelvin = KelvinFilterXForm()
        self.lomo = LomoFilterXForm()
        self.toaster = ToasterXForm()

    def test_data_integrity(self):
        start_array = self.rgb_entity.get_data()
        end_array = self.noop.do(self.rgb_entity, self.random_state).get_data()
        self.assertTrue(np.array_equal(start_array, end_array))
        start_array = self.rgba_entity.get_data()
        end_array = self.noop.do(self.rgba_entity, self.random_state).get_data()
        self.assertTrue(np.array_equal(start_array, end_array))
        start_array = self.rgba_entity.get_data()
        end_array = self.noop_down.do(self.rgba_entity, self.random_state).get_data()
        self.assertTrue(np.array_equal(start_array[:, :, :3], end_array))

    def test_gotham(self):
        out_rgb = self.gotham.do(self.rgb_entity, self.random_state)
        self.assertEqual(3, out_rgb.get_data().shape[2])
        out_rgba = self.gotham.do(self.rgba_entity, self.random_state)
        self.assertEqual(4, out_rgba.get_data().shape[2])

    def test_nashville(self):
        out_rgb = self.nashville.do(self.rgb_entity, self.random_state)
        self.assertEqual(3, out_rgb.get_data().shape[2])
        out_rgba = self.nashville.do(self.rgba_entity, self.random_state)
        self.assertEqual(4, out_rgba.get_data().shape[2])

    def test_kelvin(self):
        out_rgb = self.kelvin.do(self.rgb_entity, self.random_state)
        self.assertEqual(3, out_rgb.get_data().shape[2])
        out_rgba = self.kelvin.do(self.rgba_entity, self.random_state)
        self.assertEqual(4, out_rgba.get_data().shape[2])

    def test_lomo(self):
        out_rgb = self.lomo.do(self.rgb_entity, self.random_state)
        self.assertEqual(3, out_rgb.get_data().shape[2])
        out_rgba = self.lomo.do(self.rgba_entity, self.random_state)
        self.assertEqual(4, out_rgba.get_data().shape[2])

    def test_toaster(self):
        out_rgb = self.toaster.do(self.rgb_entity, self.random_state)
        self.assertEqual(3, out_rgb.get_data().shape[2])
        out_rgba = self.toaster.do(self.rgba_entity, self.random_state)
        self.assertEqual(4, out_rgba.get_data().shape[2])

    def test_channel_order(self):
        bgr_lomo = LomoFilterXForm('BGR')
        rgb_lomo = LomoFilterXForm('RGB')
        bgr_img = np.concatenate((np.ones((5, 5, 1)), np.zeros((5, 5, 2))), axis=2).astype(np.uint8)
        rgb_img = np.concatenate((np.zeros((5, 5, 2)), np.ones((5, 5, 1))), axis=2).astype(np.uint8)
        bgr_result = bgr_lomo.do(GenericImageEntity(bgr_img), random_state_obj=self.random_state)
        rgb_result = rgb_lomo.do(GenericImageEntity(rgb_img), random_state_obj=self.random_state)
        self.assertTrue(np.array_equal(bgr_result.get_data()[:, :, 0], rgb_result.get_data()[:, :, 2]))
        bgr_switched_result = rgb_lomo.do(GenericImageEntity(bgr_img), random_state_obj=self.random_state)
        rgb_switched_result = bgr_lomo.do(GenericImageEntity(rgb_img), random_state_obj=self.random_state)
        # transform should be modifying R and G channels, but is instead modifying B and G channels, setting to zero
        self.assertTrue(np.array_equal(bgr_switched_result.get_data(), np.zeros((5, 5, 3))))
        self.assertTrue(np.array_equal(rgb_switched_result.get_data(), np.zeros((5, 5, 3))))


if __name__ == '__main__':
    unittest.main()
