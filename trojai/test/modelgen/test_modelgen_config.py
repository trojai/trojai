import unittest
from unittest.mock import Mock

import os
import shutil
import tempfile

import torchvision.models as models

from trojai.modelgen.architecture_factory import ArchitectureFactory
from trojai.modelgen.data_manager import DataManager
from trojai.modelgen.config import ModelGeneratorConfig


class MyArchFactory(ArchitectureFactory):
    def new_architecture(self):
        return models.alexnet()


class TestModelGeneratorConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.m_tmp_dir = tempfile.TemporaryDirectory()
        self.s_tmp_dir = tempfile.TemporaryDirectory()
        self.model_save_dir = self.m_tmp_dir.name
        self.stats_save_dir = self.s_tmp_dir.name
        self.tdm = Mock(spec=DataManager)

    def tearDown(self):
        self.m_tmp_dir.cleanup()
        self.s_tmp_dir.cleanup()

    def test_good_param_configs(self):
        mgc = ModelGeneratorConfig(MyArchFactory(), self.tdm, self.model_save_dir, self.stats_save_dir, 10)
        self.assertIsInstance(mgc.arch_factory, ArchitectureFactory)
        self.assertIsInstance(mgc.data, DataManager)
        self.assertEqual(mgc.data, self.tdm)
        self.assertEqual(mgc.model_save_dir, self.model_save_dir)
        self.assertEqual(mgc.num_models, 10)

        mgc = ModelGeneratorConfig(MyArchFactory(), self.tdm, self.model_save_dir, self.stats_save_dir, 15)

        self.assertIsInstance(mgc.arch_factory, ArchitectureFactory)
        self.assertIsInstance(mgc.data, DataManager)
        self.assertEqual(mgc.data, self.tdm)
        self.assertEqual(mgc.model_save_dir, self.model_save_dir)
        self.assertEqual(mgc.num_models, 15)

    def test_arch_and_data_bad_args(self):
        self.assertRaises(TypeError, ModelGeneratorConfig, 5, self.tdm, self.model_save_dir, 10)
        self.assertRaises(TypeError, ModelGeneratorConfig, MyArchFactory(), '5', self.model_save_dir, 10)

    def test_model_save_dir_bad_args(self):
        # error is the arg 5
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, 5, 10)
        # error is arg 'object'
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, object, 10)

    def test_stats_save_dir_bad_args(self):
        # error is the arg 5
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, 5, 10)
        # error is arg 'object'
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, object, 10)

    def test_num_models_bad_args(self):
        # error is arg 'object'
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, self.model_save_dir, self.stats_save_dir, object)
        # error is arg '1'
        with self.assertRaises(TypeError):
            ModelGeneratorConfig(MyArchFactory(), self.tdm, self.model_save_dir, self.stats_save_dir, '1')


if __name__ == "__main__":
    unittest.main()
