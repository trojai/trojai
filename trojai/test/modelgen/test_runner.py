import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import torch.nn as nn
import torchvision.models as models

from trojai.modelgen.architecture_factory import ArchitectureFactory
from trojai.modelgen.config import RunnerConfig, TrainingConfig, DefaultOptimizerConfig
from trojai.modelgen.data_manager import DataManager
from trojai.modelgen.default_optimizer import DefaultOptimizer
from trojai.modelgen.optimizer_interface import OptimizerInterface
from trojai.modelgen.runner import Runner, add_numerical_extension
from trojai.modelgen.training_statistics import BatchStatistics, EpochStatistics
from trojai.modelgen.training_statistics import TrainingRunStatistics

warnings.filterwarnings("ignore")


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_runner_init_good_arg(self):
        mock_runner_config = Mock(spec=RunnerConfig)
        runner = Runner(mock_runner_config)
        self.assertEqual(runner.cfg, mock_runner_config)

    def test_runner_init_bad_arg(self):
        self.assertRaises(TypeError, Runner, object())

    def test_run(self):
        # set up functions and values that will be called and returned in run function
        mock_runner_config = Mock(spec=RunnerConfig)
        train_mock = Mock()

        train = (train_mock for _ in range(1))
        ctest, ttest, cttest = Mock(), Mock(), Mock()
        dd1, dd2, dd3, dd4 = Mock(), Mock(), Mock(), Mock()  # the data description classes
        mock_runner_config.data = Mock(spec=DataManager)
        mock_runner_config.data.load_data = Mock()
        mock_runner_config.data.load_data.return_value = (train, ctest, ttest, cttest, dd1, dd2, dd3, dd4)
        mock_runner_config.data.train_dataloader_kwargs = None
        mock_runner_config.data.test_dataloader_kwargs = None
        mock_runner_config.optimizer = Mock(spec=OptimizerInterface)
        mock_optimizer1 = Mock(spec=DefaultOptimizer)
        mock_optimizer1.train = Mock()
        mock_runner_config.optimizer_generator = (opt for opt in [mock_optimizer1])
        mock_runner_config.arch_factory = Mock(spec=ArchitectureFactory)
        mock_runner_config.arch_factory.new_architecture = Mock()
        arch = Mock(spec=nn.Module)
        mock_runner_config.arch_factory.new_architecture.return_value = arch
        mock_runner_config.arch_factory_kwargs = None
        mock_runner_config.arch_factory_kwargs_generator = None
        mock_runner_config.parallel = False
        mock_runner_config.amp = False

        test_batch_stats = BatchStatistics(1, 1, 1)
        e = EpochStatistics(1)
        e.add_batch(test_batch_stats)

        mock_optimizer1.train.return_value = (arch, [e], 1, -1)
        mock_optimizer1.test = Mock()
        mock_training_cfg1 = Mock(spec=TrainingConfig)
        mock_optimizer1.get_cfg_as_dict.return_value = mock_training_cfg1

        test_return_dict = dict()
        test_return_dict['clean_accuracy'] = 1
        test_return_dict['clean_n_total'] = 1
        test_return_dict['triggered_accuracy'] = 1
        test_return_dict['triggered_n_total'] = 1
        mock_optimizer1.test.return_value = test_return_dict

        # create runner and set mock _save_model function
        runner = Runner(mock_runner_config)
        mock_save_model = Mock()
        runner._save_model_and_stats = mock_save_model

        # run function
        with patch("trojai.modelgen.runner.TrainingRunStatistics") as p:
            runner.run()

            # check if correct functions were called with correct arguments and the correct number of times
            mock_runner_config.data.load_data.assert_called_once_with()
            mock_optimizer1.train.assert_called_once_with(arch, train_mock, None, use_amp=mock_runner_config.amp)
            mock_optimizer1.test.assert_called_once_with(arch, ctest, ttest, cttest, None)
            mock_save_model.assert_called_once_with(arch, p(), [mock_training_cfg1])

    def test_run_with_iterable_data(self):
        # set up functions and values that will be called and returned in run function
        mock_runner_config = Mock(spec=RunnerConfig)
        train1, train2, train3 = Mock(), Mock(), Mock()

        train = (t for t in [train1, train2, train3])
        ctest, ttest, cttest = Mock(), Mock(), Mock()
        dd1, dd2, dd3, dd4 = Mock(), Mock(), Mock(), Mock()  # the data descriptor classes
        mock_runner_config.data = Mock(spec=DataManager)
        mock_runner_config.data.torch_dataloader_kwargs = None
        mock_runner_config.data.iterable_training = True
        mock_runner_config.data.load_data = Mock()
        mock_runner_config.data.load_data.return_value = (train, ctest, ttest, cttest, dd1, dd2, dd3, dd4)
        mock_runner_config.data.train_dataloader_kwargs = None
        mock_runner_config.data.test_dataloader_kwargs = None
        mock_runner_config.arch_factory = Mock(spec=ArchitectureFactory)
        mock_runner_config.arch_factory.new_architecture = Mock()
        arch = Mock(spec=nn.Module)
        mock_runner_config.arch_factory.new_architecture.return_value = arch
        mock_runner_config.arch_factory_kwargs = None
        mock_runner_config.arch_factory_kwargs_generator = None
        mock_runner_config.optimizer = Mock(spec=OptimizerInterface)
        mock_optimizer1 = Mock(spec=DefaultOptimizer)
        mock_optimizer1.train = Mock()
        mock_training_cfg1 = Mock(spec=TrainingConfig)
        mock_optimizer1.get_cfg_as_dict.return_value = mock_training_cfg1
        mock_runner_config.optimizer_generator = (mock_optimizer1 for _ in range(3))
        mock_runner_config.parallel = False
        mock_runner_config.amp = False

        test_batch_stats = BatchStatistics(1, 1, 1)
        e = EpochStatistics(1)
        e.add_batch(test_batch_stats)

        mock_optimizer1.train.return_value = (arch, [e], 1, -1)
        mock_optimizer1.test = Mock()

        test_return_dict = dict()
        test_return_dict['clean_accuracy'] = 1
        test_return_dict['clean_n_total'] = 1
        test_return_dict['triggered_accuracy'] = 1
        test_return_dict['triggered_n_total'] = 1
        mock_optimizer1.test.return_value = test_return_dict

        # create runner and set mock _save_model function
        runner = Runner(mock_runner_config)
        mock_save_model = Mock()
        runner._save_model_and_stats = mock_save_model

        calls = [unittest.mock.call(arch, train1, None, use_amp=mock_runner_config.amp),
                 unittest.mock.call(arch, train2, None, use_amp=mock_runner_config.amp),
                 unittest.mock.call(arch, train3, None, use_amp=mock_runner_config.amp)]

        # run function
        with patch("trojai.modelgen.runner.TrainingRunStatistics") as p:
            runner.run()

            # check if correct functions were called with correct arguments and the correct number of times
            mock_runner_config.data.load_data.assert_called_once_with()
            mock_optimizer1.train.assert_has_calls(calls, any_order=False)
            mock_optimizer1.test.assert_called_once_with(arch, ctest, ttest, cttest, None)
            mock_save_model.assert_called_once_with(arch, p(), [mock_training_cfg1, mock_training_cfg1,
                                                                mock_training_cfg1])

        # again with multiple optimizers
        mock_runner_config = Mock(spec=RunnerConfig)
        train1, train2, train3 = Mock(), Mock(), Mock()

        train = (t for t in [train1, train2, train3])
        ctest, ttest, cttest = Mock(), Mock(), Mock()
        dd1, dd2, dd3, dd4 = Mock(), Mock(), Mock(), Mock()  # the data descriptor classes
        mock_runner_config.data = Mock(spec=DataManager)
        mock_runner_config.data.torch_dataloader_kwargs = None
        mock_runner_config.data.iterable_training = True
        mock_runner_config.data.load_data = Mock()
        mock_runner_config.data.load_data.return_value = (train, ctest, ttest, cttest, dd1, dd2, dd3, dd4)
        mock_runner_config.data.train_dataloader_kwargs = None
        mock_runner_config.data.test_dataloader_kwargs = None
        mock_runner_config.arch_factory = Mock(spec=ArchitectureFactory)
        mock_runner_config.arch_factory.new_architecture = Mock()
        arch = Mock(spec=nn.Module)
        mock_runner_config.arch_factory.new_architecture.return_value = arch
        mock_runner_config.arch_factory_kwargs = None
        mock_runner_config.arch_factory_kwargs_generator = None
        mock_runner_config.parallel = False
        mock_runner_config.amp = False

        mock_runner_config.optimizer = Mock(spec=OptimizerInterface)
        mock_optimizer1 = Mock(spec=DefaultOptimizer)
        mock_optimizer1.train = Mock()
        mock_training_cfg1 = Mock(spec=TrainingConfig)
        mock_optimizer1.get_cfg_as_dict.return_value = mock_training_cfg1
        mock_optimizer2 = Mock(spec=DefaultOptimizer)
        mock_optimizer2.train = Mock()
        mock_training_cfg2 = Mock(spec=TrainingConfig)
        mock_optimizer2.get_cfg_as_dict.return_value = mock_training_cfg2
        mock_optimizer3 = Mock(spec=DefaultOptimizer)
        mock_optimizer3.train = Mock()
        mock_training_cfg3 = Mock(spec=TrainingConfig)
        mock_optimizer3.get_cfg_as_dict.return_value = mock_training_cfg3
        mock_runner_config.optimizer_generator = (mo for mo in [mock_optimizer1, mock_optimizer2, mock_optimizer3])

        test_batch_stats = BatchStatistics(1, 1, 1)
        e = EpochStatistics(1)
        e.add_batch(test_batch_stats)

        mock_optimizer1.train.return_value = (arch, [e], 1, -1)
        mock_optimizer1.test = Mock()
        mock_optimizer2.train.return_value = (arch, [e], 1, -1)
        mock_optimizer2.test = Mock()
        mock_optimizer3.train.return_value = (arch, [e], 1, -1)
        mock_optimizer3.test = Mock()

        test_return_dict = dict()
        test_return_dict['clean_accuracy'] = 1
        test_return_dict['clean_n_total'] = 1
        test_return_dict['triggered_accuracy'] = 1
        test_return_dict['triggered_n_total'] = 1
        mock_optimizer3.test.return_value = test_return_dict

        # create runner and set mock _save_model function
        runner = Runner(mock_runner_config)
        mock_save_model = Mock()
        runner._save_model_and_stats = mock_save_model

        # run function
        with patch("trojai.modelgen.runner.TrainingRunStatistics") as p:
            runner.run()

            mock_optimizer1.train.assert_called_once_with(arch, train1, None, use_amp=mock_runner_config.amp)
            mock_optimizer1.test.assert_not_called()
            mock_optimizer2.train.assert_called_once_with(arch, train2, None, use_amp=mock_runner_config.amp)
            mock_optimizer2.test.assert_not_called()
            mock_optimizer3.train.assert_called_once_with(arch, train3, None, use_amp=mock_runner_config.amp)
            mock_optimizer3.test.assert_called_once_with(arch, ctest, ttest, cttest, None)

    def test_get_training_cfg(self):
        mock_default_optimizer_cfg = Mock(spec=DefaultOptimizerConfig)
        mock_default_optimizer = Mock(spec=DefaultOptimizer)
        mock_optimizer_interface = Mock(spec=OptimizerInterface)

        mock_default_optimizer_cfg.training_cfg = Mock()

        mock_default_optimizer_cfg.training_cfg.get_cfg_as_dict = Mock(return_value={'default_cfg': True, '1': 1})
        mock_default_optimizer.get_cfg_as_dict = Mock(return_value={'default_opt': True, '2': 2})
        mock_optimizer_interface.get_cfg_as_dict = Mock(return_value={'opt_inter': True, '3': 3})

        self.assertEqual(Runner._get_training_cfg(mock_default_optimizer_cfg), {'default_cfg': True, '1': 1})
        self.assertEqual(Runner._get_training_cfg(mock_default_optimizer), {'default_opt': True, '2': 2})
        self.assertEqual(Runner._get_training_cfg(mock_optimizer_interface), {'opt_inter': True, '3': 3})
        self.assertEqual(Runner._get_training_cfg(object()), {})

    def test_save_model(self):
        tmp_1 = tempfile.TemporaryDirectory()
        stats_tmp_1 = tempfile.TemporaryDirectory()

        # path and file names to test to ensure correct file name saving
        path1 = tmp_1.name
        path1_stats_dir = stats_tmp_1.name
        path2 = path1 + '/'
        path2_stats_dir = path1_stats_dir + '/'
        pt_filename = 'model.pt'

        m_model = models.alexnet()
        ts = TrainingRunStatistics()

        mock_runner_config = Mock(spec=RunnerConfig)
        mock_runner_config.parallel = False
        runner = Runner(mock_runner_config)

        mock_runner_config.model_save_dir = path1
        mock_runner_config.stats_save_dir = path1_stats_dir
        mock_runner_config.filename = pt_filename
        mock_runner_config.run_id = None
        mock_runner_config.save_with_hash = False
        mock_runner_config.model_save_format = 'pt'

        mock_runner_config.optimizer = DefaultOptimizer()
        runner._save_model_and_stats(m_model, ts, [])
        self.assertTrue(os.path.isfile(path2 + pt_filename + '.1'))

        mock_runner_config.model_save_dir = path2
        mock_runner_config.stats_save_dir = path2_stats_dir
        runner._save_model_and_stats(m_model, ts, [])
        self.assertTrue(os.path.isfile(path2 + 'model.pt.2'))

        mock_runner_config.filename = None
        mock_runner_config.run_id = 50
        runner._save_model_and_stats(m_model, ts, [])
        self.assertTrue(os.path.isfile(path2 + 'AlexNet_id50.pt.1'))

        tmp_1.cleanup()
        stats_tmp_1.cleanup()

    def test_add_numerical_extension(self):
        with tempfile.TemporaryDirectory() as p:
            input_filenames = ['model.pt', 'model.alpha_0.2.pt', 'model.alpha_0.2.pt.1']
            expected_output_filenames = ['model.pt.1', 'model.alpha_0.2.pt.1', 'model.alpha_0.2.pt.2']
            for ii in range(len(input_filenames)):
                input_fname = input_filenames[ii]
                expected_output_fname = expected_output_filenames[ii]
                actual_output_fname = add_numerical_extension(p, input_fname)
                self.assertEqual(expected_output_fname, actual_output_fname)

            # check when files actually exist on disk in a serial fashion
            query_fname = 'model.pt'
            for ii in range(25):
                expected_output_fname = query_fname+'.'+str(ii+1)  # b/c ii starts at 0
                actual_output_fname = add_numerical_extension(p, query_fname)
                self.assertEqual(expected_output_fname, actual_output_fname)
                # now write this file to disk, to ensure the globbing works properly and we increment to the following
                # digit
                Path(os.path.join(p, actual_output_fname)).touch()

    def test_add_numerical_extension2(self):
        with tempfile.TemporaryDirectory() as p:
            fname_prefix = 'ModdedLeNet5Net_mnist_dynamic_norotation__alphatrigger_0.05.pt'
            num_models = 10
            for ii in range(1, num_models+1):
                Path(os.path.join(p, fname_prefix+'.'+str(ii))).touch()
            next_actual_fname = add_numerical_extension(p, fname_prefix)
            next_expected_fname = fname_prefix+'.11'
            self.assertEqual(next_expected_fname, next_actual_fname)

    def test_add_numerical_extension3(self):
        with tempfile.TemporaryDirectory() as p:
            fname_prefix = 'model.pt'
            num_models = 10
            for ii in range(1, num_models+1):
                # create model skeleton
                Path(os.path.join(p, fname_prefix+'.'+str(ii))).touch()
                # create stats skeleton
                Path(os.path.join(p, fname_prefix + '.' + str(ii)+'.stats.json')).touch()
                # create detailed stats skeleton
                Path(os.path.join(p, fname_prefix + '.' + str(ii) + '.stats.detailed.csv')).touch()
            next_actual_model_fname = add_numerical_extension(p, fname_prefix)
            next_expected_model_fname = fname_prefix+'.11'
            self.assertEqual(next_expected_model_fname, next_actual_model_fname)


if __name__ == "__main__":
    unittest.main()
