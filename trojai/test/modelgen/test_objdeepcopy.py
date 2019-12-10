import unittest
import copy
import torch
import os
import shutil
import csv
import tempfile

import trojai.modelgen.config as tpmc
import trojai.modelgen.architecture_factory as tpmaf
import trojai.modelgen.data_manager as tpmd
import trojai.modelgen.architectures.mnist_architectures as tpma
import trojai.modelgen.default_optimizer as tpmo

"""
Test custom __deepcopy__ implementations
"""


class TestCopyImplementations(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryDirectory()
        self.experiment_path = self.tmp_file.name
        self.train_file = os.path.join(self.experiment_path, "train.csv")
        self.clean_test_file = os.path.join(self.experiment_path, "test.csv")
        self.triggered_file = os.path.join(self.experiment_path, "triggered.csv")
        self.model_save_dir = os.path.join(self.experiment_path, "model_save_dir")
        self.stats_save_dir = os.path.join(self.experiment_path, "stats_save_dir")

        # make dummy CSV files to pass validation()
        dummy_csv_files = [self.train_file, self.clean_test_file, self.triggered_file]
        for ff in dummy_csv_files:
            dummy_dict = {"a": "1", "b": "2", "c": "3"}
            with open(ff, "w") as f:
                writer = csv.writer(f)
                for i in dummy_dict:
                    writer.writerow([i, dummy_dict[i]])
            f.close()

    def tearDown(self) -> None:
        self.tmp_file.cleanup()

    def test_training_config_copy1(self):
        t1 = tpmc.TrainingConfig()
        t2 = copy.deepcopy(t1)
        self.assertEqual(t1, t2)

    def test_training_config_copy2(self):
        t1 = tpmc.TrainingConfig(torch.device("cpu"))
        t2 = copy.deepcopy(t1)
        self.assertEqual(t1, t2)

    def test_reporting_config_copy(self):
        r1 = tpmc.ReportingConfig()
        r2 = copy.deepcopy(r1)
        self.assertEqual(r1, r2)

    def test_default_optimizer_config_copy(self):
        o1 = tpmc.DefaultOptimizerConfig()
        o2 = tpmc.DefaultOptimizerConfig()
        self.assertEqual(o1, o2)

    def test_model_generator_config_copy(self):
        class MyArchFactory(tpmaf.ArchitectureFactory):
            def new_architecture(self):
                return tpma.ModdedLeNet5Net(channels=1)
        arch = MyArchFactory()
        # setup the xforms to ensure we can test the callables
        def data_xform(x): return x*x
        def label_xform(y): return y*y*y
        data = tpmd.DataManager(self.experiment_path, self.train_file, self.clean_test_file,
                                triggered_test_file=self.triggered_file,
                                train_data_transform=data_xform,
                                train_label_transform=label_xform,
                                test_data_transform=data_xform,
                                test_label_transform=label_xform,
                                file_loader='image',
                                shuffle_train=True,
                                shuffle_clean_test=False,
                                shuffle_triggered_test=False)
        num_models = 1
        mgc1 = tpmc.ModelGeneratorConfig(arch, data, self.model_save_dir, self.stats_save_dir, num_models)
        mgc2 = copy.deepcopy(mgc1)
        self.assertEqual(mgc1, mgc2)

    def test_default_optimizer_copy(self):
        opt1 = tpmo.DefaultOptimizer()
        opt2 = tpmo.DefaultOptimizer()
        self.assertEqual(opt1, opt2)

    def test_data_manager_copy(self):
        def train_data_xform(x): return x*x
        def train_label_xform(y): return y*y*y
        def test_data_xform(x): return x**2
        def test_label_xform(y): return y + 2
        dat1 = tpmd.DataManager(self.experiment_path, self.train_file, self.clean_test_file,
                                triggered_test_file=self.triggered_file,
                                train_data_transform=train_data_xform,
                                train_label_transform=train_label_xform,
                                test_data_transform=test_data_xform,
                                test_label_transform=test_label_xform,
                                file_loader='image',
                                shuffle_train=True,
                                shuffle_clean_test=False,
                                shuffle_triggered_test=False)
        dat2 = copy.deepcopy(dat1)
        self.assertEqual(dat1, dat2)
        self.assertEqual(train_data_xform, dat1.train_data_transform)
        self.assertEqual(train_data_xform, dat2.train_data_transform)
        self.assertEqual(train_label_xform, dat1.train_label_transform)
        self.assertEqual(train_label_xform, dat2.train_label_transform)
        self.assertEqual(test_data_xform, dat1.test_data_transform)
        self.assertEqual(test_data_xform, dat2.test_data_transform)
        self.assertEqual(test_label_xform, dat1.test_label_transform)
        self.assertEqual(test_label_xform, dat2.test_label_transform)


if __name__ == '__main__':
    unittest.main()
