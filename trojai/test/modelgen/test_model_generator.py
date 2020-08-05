import unittest
from unittest.mock import patch, Mock

import torch
import torch.nn

from trojai.modelgen.model_generator import ModelGenerator
from trojai.modelgen.architecture_factory import ArchitectureFactory
from trojai.modelgen.data_manager import DataManager
from trojai.modelgen.optimizer_interface import OptimizerInterface
from trojai.modelgen.config import TrainingConfig, ReportingConfig, ModelGeneratorConfig


class TestModelGenerator(unittest.TestCase):
    def test_good_init(self):
        mgc_mock1 = Mock(spec=ModelGeneratorConfig)
        mgc_mock2 = Mock(spec=ModelGeneratorConfig)
        mg = ModelGenerator(mgc_mock1)
        self.assertEqual(mg.configs[0], mgc_mock1)
        cfgs = [mgc_mock1, mgc_mock2]
        mg = ModelGenerator(cfgs)
        self.assertEqual(mg.configs, cfgs)

    def test_bad_init(self):
        self.assertRaises(TypeError, ModelGenerator, 1)
        self.assertRaises(RuntimeError, ModelGenerator, [])
        self.assertRaises(TypeError, ModelGenerator, [0])

    def test_run_config_single(self):
        mgc_mock1 = Mock(spec=ModelGeneratorConfig)
        mgc_mock1.arch_factory = Mock(spec=ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        mgc_mock1.arch_factory.new_architecture.return_value = arch
        mgc_mock1.data = Mock(spec=DataManager)
        mgc_mock1.data.train_file = 'a'
        mgc_mock1.model_save_dir = './test_dir/'
        mgc_mock1.stats_save_dir = './test_stats_dir/'
        mgc_mock1.optimizer = Mock(spec=OptimizerInterface)
        mgc_mock1.num_models = 1
        mgc_mock1.arch_factory_kwargs = None
        mgc_mock1.arch_factory_kwargs_generator = None
        mgc_mock1.parallel = False
        mgc_mock1.objective = None

        training_params = TrainingConfig(optim=mgc_mock1.optimizer)
        logging_params = ReportingConfig()
        mgc_mock1.training_cfg = training_params
        mgc_mock1.reporting_cfg = logging_params
        mgc_mock1.experiment_cfg = dict(name='test')
        mgc_mock1.save_with_hash = False

        mgc_mock1.run_ids = None
        mgc_mock1.filenames = None
        mg = ModelGenerator(mgc_mock1)

        run_ret = Mock()
        with patch('trojai.modelgen.model_generator.modelgen_cfg_to_runner_cfg') as rc:
            with patch('trojai.modelgen.model_generator.Runner', Mock(return_value=run_ret)):
                mg.run()
                rc.assert_called_once_with(mgc_mock1, run_id=None, filename=None)
                run_ret.run.assert_called_once_with()

    def test_run_config_list(self):
        mgc_mock1 = Mock(spec=ModelGeneratorConfig)
        mgc_mock1.arch_factory = Mock(spec=ArchitectureFactory)
        mgc_mock1.arch_factory_kwargs = {}
        mgc_mock1.arch_factory_kwargs_generator = None
        arch = Mock(spec=torch.nn.Module)
        mgc_mock1.arch_factory.new_architecture.return_value = arch
        mgc_mock1.data = Mock(spec=DataManager)
        mgc_mock1.data.train_file = 'a'
        mgc_mock1.model_save_dir = './test_dir/'
        mgc_mock1.stats_save_dir = './test_stats_dir/'
        mgc_mock1.optimizer = Mock(spec=OptimizerInterface)
        mgc_mock1.num_models = 1
        mgc_mock1.parallel = False
        mgc_mock1.train_val_split = 0.
        mgc_mock1.training_cfg = None
        mgc_mock1.reporting_cfg = None
        mgc_mock1.run_ids = None
        mgc_mock1.filenames = None

        training_params = TrainingConfig(optim=mgc_mock1.optimizer)
        logging_params = ReportingConfig()

        mgc_mock1.training_cfg = training_params
        mgc_mock1.reporting_cfg = logging_params
        mgc_mock1.experiment_cfg = dict(name='test')

        mgc_mock2 = Mock(spec=ModelGeneratorConfig)
        mgc_mock2.arch_factory = Mock(spec=ArchitectureFactory)
        mgc_mock2.arch_factory_kwargs = {}
        mgc_mock2.arch_factory_kwargs_generator = None
        arch2 = Mock(spec=torch.nn.Module)
        mgc_mock2.arch_factory.new_architecture.return_value = arch2
        mgc_mock2.data = Mock(spec=DataManager)
        mgc_mock2.data.train_file = 'b'
        mgc_mock2.optimizer = mgc_mock1.optimizer
        mgc_mock2.model_save_dir = './test_dir/'
        mgc_mock2.stats_save_dir = './test_stats_dir/'
        mgc_mock2.num_models = 2
        mgc_mock2.parallel = False
        mgc_mock2.train_val_split = 0.
        mgc_mock2.training_cfg = training_params
        mgc_mock2.reporting_cfg = logging_params
        mgc_mock2.experiment_cfg = dict(name='test')

        mgc_mock1.amp = False
        mgc_mock2.amp = False

        mgc_mock1.save_with_hash = False
        mgc_mock2.save_with_hash = False

        mgc_mock2.run_ids = None
        mgc_mock2.filenames = ['1', '2']

        mg = ModelGenerator([mgc_mock1, mgc_mock2])
        run_ret = Mock()
# <<<<<<< HEAD
        with patch('trojai.modelgen.model_generator.Runner', Mock(return_value=run_ret)):
            mg.run()
            self.assertEqual(run_ret.run.call_count, 3)

            # TODO: what's going on here?
# =======
#         calls = [unittest.mock.call(mgc_mock1, run_id=None, filename=None),
#                  unittest.mock.call(mgc_mock2, run_id=None, filename='1'),
#                  unittest.mock.call(mgc_mock2, run_id=None, filename='2')]
#         with patch('trojai.modelgen.single_threaded_model_generator.modelgen_cfg_to_runner_cfg') as rc:
#             with patch('trojai.modelgen.single_threaded_model_generator.Runner', Mock(return_value=run_ret)):
#                 mg.run()
#                 self.assertEqual(run_ret.run.call_count, 3)
#                 rc.assert_has_calls(calls)
# >>>>>>> master


if __name__ == "__main__":
    unittest.main()
