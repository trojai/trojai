import unittest
from unittest.mock import Mock

import torch.nn
import os
import tempfile

import trojai.modelgen.config
import trojai.modelgen.uge_model_generator as tpmu
import trojai.modelgen.config as tpmc
import trojai.modelgen.architecture_factory as tpmaf
import trojai.modelgen.data_manager as tpmdm
import trojai.modelgen.optimizer_interface as tpmo


class TestUGEModelGenerator(unittest.TestCase):
    """
    Tests the UGE Model Generator
    """
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.uge_working_directory = os.path.join(self.tmp_dir.name, "uge_wd_1")
        self.model_save_dir_cfg1 = os.path.join(self.tmp_dir.name, "test_models_dir1")
        self.stats_save_dir_cfg1 = os.path.join(self.tmp_dir.name, "test_stats_dir1")
        self.model_save_dir_cfg2 = os.path.join(self.tmp_dir.name, "test_models_dir2")
        self.stats_save_dir_cfg2 = os.path.join(self.tmp_dir.name, "test_stats_dir2")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_expand_modelgen_configs1(self):
        q1 = trojai.modelgen.config.UGEQueueConfig("gpu-k40.q", True)
        uge_config = trojai.modelgen.config.UGEConfig(q1, None)

        # setup a fake ModelGeneratorConfig object
        modelgen_cfg1 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg1.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg1.arch_factory.new_architecture.return_value = arch
        modelgen_cfg1.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg1.model_save_dir = self.model_save_dir_cfg1
        modelgen_cfg1.stats_save_dir = self.stats_save_dir_cfg1
        modelgen_cfg1.num_models = 3
        modelgen_cfg1.optimizer = Mock(spec=tpmo.OptimizerInterface)
        mg = tpmu.UGEModelGenerator(modelgen_cfg1, uge_config,
                                    working_directory=self.uge_working_directory,
                                    validate_uge_dirs=False)

        expanded_configs = mg.expand_modelgen_configs_to_process()
        self.assertEqual(len(expanded_configs), modelgen_cfg1.num_models)
        for ii in range(len(expanded_configs)):
            self.assertEqual(expanded_configs[ii].num_models, 1)

    def test_expand_modelgen_configs2(self):
        q1 = trojai.modelgen.config.UGEQueueConfig("gpu-k40.q", True)
        uge_config = trojai.modelgen.config.UGEConfig(q1, None)

        # setup a fake ModelGeneratorConfig object
        modelgen_cfg1 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg1.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg1.arch_factory.new_architecture.return_value = arch
        modelgen_cfg1.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg1.model_save_dir = self.model_save_dir_cfg1
        modelgen_cfg1.stats_save_dir = self.stats_save_dir_cfg1
        modelgen_cfg1.num_models = 1
        modelgen_cfg1.optimizer = Mock(spec=tpmo.OptimizerInterface)

        modelgen_cfg2 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg2.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg2.arch_factory.new_architecture.return_value = arch
        modelgen_cfg2.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg2.model_save_dir = self.model_save_dir_cfg2
        modelgen_cfg2.stats_save_dir = self.stats_save_dir_cfg2
        modelgen_cfg2.num_models = 1
        modelgen_cfg2.optimizer = Mock(spec=tpmo.OptimizerInterface)

        modelgen_cfgs_to_process = [modelgen_cfg1, modelgen_cfg2]

        mg = tpmu.UGEModelGenerator(modelgen_cfgs_to_process, uge_config,
                                    working_directory=self.uge_working_directory,
                                    validate_uge_dirs=False)

        expanded_configs = mg.expand_modelgen_configs_to_process()
        self.assertEqual(len(expanded_configs), len(modelgen_cfgs_to_process))

    def test_expand_modelgen_configs3(self):
        q1 = trojai.modelgen.config.UGEQueueConfig("gpu-k40.q", True)
        uge_config = trojai.modelgen.config.UGEConfig(q1, None)

        # setup a fake ModelGeneratorConfig object
        modelgen_cfg1 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg1.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg1.arch_factory.new_architecture.return_value = arch
        modelgen_cfg1.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg1.model_save_dir = self.model_save_dir_cfg1
        modelgen_cfg1.stats_save_dir = self.stats_save_dir_cfg1
        modelgen_cfg1.num_models = 2
        modelgen_cfg1.optimizer = Mock(spec=tpmo.OptimizerInterface)

        modelgen_cfg2 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg2.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg2.arch_factory.new_architecture.return_value = arch
        modelgen_cfg2.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg2.model_save_dir = self.model_save_dir_cfg2
        modelgen_cfg2.stats_save_dir = self.stats_save_dir_cfg2
        modelgen_cfg2.num_models = 3
        modelgen_cfg2.optimizer = Mock(spec=tpmo.OptimizerInterface)

        modelgen_cfgs_to_process = [modelgen_cfg1, modelgen_cfg2]

        mg = tpmu.UGEModelGenerator(modelgen_cfgs_to_process, uge_config,
                                    working_directory=self.uge_working_directory,
                                    validate_uge_dirs=False)

        expanded_configs = mg.expand_modelgen_configs_to_process()
        self.assertEqual(len(expanded_configs), modelgen_cfg1.num_models+modelgen_cfg2.num_models)
        for ii in range(len(expanded_configs)):
            self.assertEqual(expanded_configs[ii].num_models, 1)

    def test_job_assigmnet1(self):
        """
        Tests the scenario where we have 1 job to process on 1 queue
        :return:
        """
        q1 = trojai.modelgen.config.UGEQueueConfig("gpu-k40.q", True)
        uge_config = trojai.modelgen.config.UGEConfig(q1, None)

        # setup a fake ModelGeneratorConfig object
        modelgen_cfg1 = Mock(spec=tpmc.ModelGeneratorConfig)
        modelgen_cfg1.arch_factory = Mock(spec=tpmaf.ArchitectureFactory)
        arch = Mock(spec=torch.nn.Module)
        modelgen_cfg1.arch_factory.new_architecture.return_value = arch
        modelgen_cfg1.data = Mock(spec=tpmdm.DataManager)
        modelgen_cfg1.model_save_dir = self.model_save_dir_cfg1
        modelgen_cfg1.stats_save_dir = self.stats_save_dir_cfg1
        modelgen_cfg1.num_models = 1
        modelgen_cfg1.optimizer = Mock(spec=tpmo.OptimizerInterface)
        mg = tpmu.UGEModelGenerator(modelgen_cfg1, uge_config,
                                    working_directory=self.uge_working_directory,
                                    validate_uge_dirs=False)

        queue_numjobs_assignment = mg.get_queue_numjobs_assignment()
        self.assertTrue(len(queue_numjobs_assignment) == 1)
        # assert 1 job was assigned to the only queue available
        self.assertTrue(queue_numjobs_assignment[0][1] == 1)
        # assert that the queue reference is maintained
        self.assertEqual(queue_numjobs_assignment[0][0], q1)

    def test_pyscript_gen(self):
        # since this is calling a static function, we need to create the directories
        try:
            os.makedirs(self.uge_working_directory)
        except IOError:
            pass
        pyscript_fname = os.path.join(self.uge_working_directory, "script.py")
        modelgen_cfg_persist_fname = os.path.join(self.uge_working_directory, "model_persist.pkl")
        persist_metadata_fname = os.path.join(self.uge_working_directory, "abc")
        pyscript_log_fname = os.path.join(self.uge_working_directory, "script.py.log")
        run_id = None
        filename = None
        tpmu.UGEModelGenerator._gen_py_script(pyscript_fname, pyscript_log_fname,
                                              modelgen_cfg_persist_fname,
                                              persist_metadata_fname, run_id, filename)

        # check file contents
        expected_file_contents = '''\
#!/usr/bin/env python
import json
import logging.config
import trojai.modelgen.config as tpmc
import trojai.modelgen.runner as tpmr

# setup logger
logging.config.dictConfig({{
    'version': 1,
    'formatters': {{
        'detailed': {{
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
    }},
    'handlers': {{
        'file': {{
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '{0}/script.py.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'INFO',
        }},
    }},
    'loggers': {{
        'trojai': {{
            'handlers': ['file'],
        }},
        'trojai_private': {{
            'handlers': ['file'],
        }},
    }},
    'root': {{
        'level': 'INFO',
    }},
}})

modelgen_cfg = tpmc.ModelGeneratorConfig.load("{0}/model_persist.pkl")
with open("{0}/abc", 'r') as f:
    persist_metadata = json.load(f)
run_cfg = tpmc.modelgen_cfg_to_runner_cfg(modelgen_cfg, run_id=None, filename=None)

runner = tpmr.Runner(run_cfg, persist_metadata=persist_metadata, progress_bar_disable=True)
runner.run()
        '''.format(self.uge_working_directory)
        with open(pyscript_fname, "r") as f:
            actual_file_contents = f.readlines()
        expected_file_contents = expected_file_contents.split('\n')
        self.assertEqual(len(expected_file_contents), len(actual_file_contents))
        for ii in range(len(expected_file_contents)):
            e_line = expected_file_contents[ii]
            a_line = actual_file_contents[ii]
            self.assertEqual(e_line.strip(), a_line.strip())

    def test_bashscript_gen(self):
        # since this is calling a static function, we need to create the directories
        try:
            os.makedirs(self.uge_working_directory)
        except IOError:
            pass
        bashscript_fname = os.path.join(self.uge_working_directory, "bash_script.sh")
        pyscript_fname = os.path.join(self.uge_working_directory, "script.py")
        tpmu.UGEModelGenerator._gen_bash_script(bashscript_fname, pyscript_fname)
        expected_file_contents = '''\
#!/bin/bash
source /etc/profile.d/modules.sh
module load cuda91
# setup conda environment
. /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate trojai

python3 {}/script.py
        '''.format(self.uge_working_directory)
        with open(bashscript_fname, "r") as f:
            actual_file_contents = f.readlines()
        expected_file_contents = expected_file_contents.split('\n')
        self.assertEqual(len(expected_file_contents), len(actual_file_contents))
        for ii in range(len(expected_file_contents)):
            e_line = expected_file_contents[ii]
            a_line = actual_file_contents[ii]
            self.assertEqual(e_line.strip(), a_line.strip())

    def test_gen_bash_command1(self):
        # since this is calling a static function, we need to create the directories
        try:
            os.makedirs(self.uge_working_directory)
        except IOError:
            pass
        bashscript_fname = os.path.join(self.uge_working_directory, "bash_script.sh")
        uge_log_fname = os.path.join(self.uge_working_directory, "uge_log.txt")
        queue_name = "test.q"

        cmd = tpmu.UGEModelGenerator._gen_bash_command(bashscript_fname, uge_log_fname, queue_name,
                                                       gpu_node=False, sync_mode=False)
        expected_command = "qsub -q test.q -V -v PATH -cwd -S /bin/bash -j y " \
                           "-o {0}/uge_log.txt {0}/bash_script.sh".format(self.uge_working_directory)
        self.assertEqual(cmd, expected_command)

    def test_gen_bash_command2(self):
        # since this is calling a static function, we need to create the directories
        bashscript_fname = os.path.join(self.uge_working_directory, "bash_script.sh")
        uge_log_fname = os.path.join(self.uge_working_directory, "uge_log.txt")
        queue_name = "test.q"

        cmd = tpmu.UGEModelGenerator._gen_bash_command(bashscript_fname, uge_log_fname, queue_name,
                                                       gpu_node=True, sync_mode=False)
        expected_command = "qsub -q test.q -l gpu=1 -V -v PATH -cwd -S /bin/bash -j y " \
                           "-o {0}/uge_log.txt {0}/bash_script.sh".format(self.uge_working_directory)
        self.assertEqual(expected_command, cmd)


if __name__ == '__main__':
    unittest.main()
