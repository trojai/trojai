import unittest
import tempfile
from unittest.mock import Mock

import torch.nn as nn

from trojai.modelgen.data_manager import DataManager
from trojai.modelgen.optimizer_interface import OptimizerInterface
from trojai.modelgen.config import RunnerConfig
from trojai.modelgen.architecture_factory import ArchitectureFactory


class TestRunnerConfig(unittest.TestCase):
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

    def tearDown(self):
        self.m_tmp_dir.cleanup()
        self.s_tmp_dir.cleanup()

    def test_good_runner_config_args(self):
        mock_arch_factory = Mock(spec=ArchitectureFactory)
        mock_data = Mock(spec=DataManager)
        mock_data.train_file = ['a', 'b']
        mock_optim = Mock(spec=OptimizerInterface)
        rc = RunnerConfig(mock_arch_factory, mock_data, None, None, mock_optim,
                          model_save_dir=self.model_save_dir, stats_save_dir=self.stats_save_dir)
        self.assertEqual(rc.arch_factory, mock_arch_factory)
        self.assertEqual(rc.data, mock_data)
        self.assertEqual(rc.optimizer, mock_optim)
        self.assertEqual(rc.model_save_dir, self.model_save_dir)

        rc = RunnerConfig(mock_arch_factory, mock_data, None, None, mock_optim,
                          model_save_dir=self.model_save_dir, stats_save_dir=self.stats_save_dir,
                          run_id=1, filename='model')
        self.assertEqual(rc.arch_factory, mock_arch_factory)
        self.assertEqual(rc.data, mock_data)
        self.assertEqual(rc.optimizer, mock_optim)
        self.assertEqual(rc.model_save_dir, self.model_save_dir)

    def test_bad_runner_config_args(self):
        mock_arch = Mock(spec=nn.Module)
        mock_data = Mock(spec=DataManager)
        mock_optim = Mock(spec=OptimizerInterface)
        self.assertRaises(TypeError, RunnerConfig, 0, mock_data, mock_optim, self.model_save_dir, self.stats_save_dir)
        self.assertRaises(TypeError, RunnerConfig, mock_arch, object(), mock_optim, self.model_save_dir,
                          self.stats_save_dir)
        self.assertRaises(TypeError, RunnerConfig, mock_arch, mock_data, 'abs', self.model_save_dir,
                          self.stats_save_dir)
        self.assertRaises(TypeError, RunnerConfig, mock_arch, mock_data, mock_optim, 2)
        self.assertRaises(TypeError, RunnerConfig, mock_arch, mock_data, mock_optim, self.model_save_dir,
                          self.stats_save_dir, filename=object())


if __name__ == "__main__":
    unittest.main()
