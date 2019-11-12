import unittest
import tempfile
import os

import pandas as pd

from trojai.modelgen.data_manager import DataManager
from trojai.modelgen.datasets import CSVDataset


def create_df(path, test_filename, empty_filename):
    df = pd.DataFrame([[1, 0, 0, 0], [0, 1, 1, 1]], [0, 1], ['col1', 'col2', 'train_label', 'true_label'])
    df.to_csv(os.path.join(path, test_filename), index=False)
    empty_df = pd.DataFrame([], [], ['col1', 'col2', 'train_label', 'true_label'])
    empty_df.to_csv(os.path.join(path, empty_filename), index=False)


class TestDataManager(unittest.TestCase):
    tmp_dir = None

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = self.tmp_dir.name
        self.file = 'test_file.csv'
        self.empty_file = "empty_file.csv"
        create_df(self.path, self.file, self.empty_file)

        self.bad_path = './bad_dir'
        self.bad_file = 'bad_file.csv'

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_good_params(self):
        # basic version
        tdm = DataManager(self.path, self.file, self.file)
        self.assertEqual(tdm.experiment_path, self.path)
        self.assertEqual(tdm.train_file, [self.file])  # data manager coverts training filename to a list
        self.assertEqual(tdm.clean_test_file, self.file)
        self.assertIsNone(tdm.triggered_test_file)
        self.assertEqual(tdm.train_data_transform(1), 1)
        self.assertEqual(tdm.train_label_transform(1), 1)
        self.assertEqual(tdm.test_data_transform(1), 1)
        self.assertEqual(tdm.test_label_transform(1), 1)
        self.assertEqual(tdm.data_loader, 'default_image_loader')
        self.assertTrue(tdm.shuffle_train)
        self.assertFalse(tdm.shuffle_clean_test)
        self.assertFalse(tdm.shuffle_triggered_test)

        # complex example
        tdm = DataManager(self.path, self.file, self.file,
                          triggered_test_file=self.file,
                          train_data_transform=lambda x: x + 1,
                          train_label_transform=lambda x: x - 1,
                          test_data_transform=lambda x: x*2,
                          test_label_transform=lambda x: x**2,
                          file_loader=lambda x: str(x),
                          shuffle_train=False,
                          shuffle_clean_test=True,
                          shuffle_triggered_test=True)

        self.assertEqual(tdm.experiment_path, self.path)
        self.assertEqual(tdm.train_file, [self.file])
        self.assertEqual(tdm.clean_test_file, self.file)
        self.assertEqual(tdm.triggered_test_file, self.file)
        self.assertEqual(tdm.train_data_transform(1), 2)
        self.assertEqual(tdm.train_label_transform(1), 0)
        self.assertEqual(tdm.test_data_transform(2), 4)
        self.assertEqual(tdm.test_label_transform(3), 9)
        self.assertEqual(tdm.data_loader(19), '19')
        self.assertFalse(tdm.shuffle_train)
        self.assertTrue(tdm.shuffle_clean_test)
        self.assertTrue(tdm.shuffle_triggered_test)

        # with multiple files given for training
        tdm = DataManager(self.path, [self.file, self.file], self.file)
        tdm = DataManager(self.path, (self.file, self.file), self.file)

    def test_load_data(self):
        tdm = DataManager(self.path, self.file, self.file,
                          triggered_test_file=self.file,
                          train_data_transform=lambda x: x + 1,
                          train_label_transform=lambda x: x - 1,
                          test_data_transform=lambda x: x + 1,
                          test_label_transform=lambda x: x - 1,
                          file_loader=lambda x: str(x),
                          shuffle_train=False,
                          shuffle_clean_test=False,
                          shuffle_triggered_test=False)
        d1, d2, d3, d4, dd1, dd2, dd3, dd4 = tdm.load_data()
        d1 = d1.__next__()
        self.assertIsInstance(d1, CSVDataset)
        self.assertIsInstance(d2, CSVDataset)
        self.assertIsInstance(d3, CSVDataset)
        self.assertIsInstance(d4, CSVDataset)
        df = pd.DataFrame([[1, 0, 0, 0], [0, 1, 1, 1]], [0, 1], ['col1', 'col2', 'train_label', 'true_label'])
        self.assertTrue(d1.data_df.equals(df))
        self.assertTrue(d2.data_df.equals(df))
        self.assertTrue(d3.data_df.equals(df))
        self.assertEqual(d1.data_loader(5), '5')
        self.assertEqual(d2.data_transform(5), 6)
        self.assertEqual(d3.label_transform(5), 4)

        # with iterable training
        tdm = DataManager(self.path, [self.file, self.file, self.file], self.file)
        d1, d2, d3, d4, dd1, dd2, dd3, dd4 = tdm.load_data()
        for d in d1:
            self.assertIsInstance(d, CSVDataset)
        self.assertIsInstance(d2, CSVDataset)
        self.assertIsNone(d3, CSVDataset)

    def test_bad_arguments(self):
        self.assertRaises((TypeError, FileNotFoundError), DataManager, 0, self.file, self.file)
        self.assertRaises((TypeError, FileNotFoundError), DataManager, self.path, object(), self.file)
        self.assertRaises((TypeError, FileNotFoundError), DataManager, self.path, self.file, 'better_raise_error')

    def test_bad_paths_and_files(self):
        self.assertRaises(FileNotFoundError, DataManager, self.bad_path, self.file, self.file)
        self.assertRaises(FileNotFoundError, DataManager, self.path, self.bad_file, self.file)
        self.assertRaises(FileNotFoundError, DataManager, self.path, self.file, self.bad_file)
        self.assertRaises(FileNotFoundError, DataManager, self.path, self.file, self.file,
                          triggered_test_file=self.bad_file)

    def test_empty_files(self):
        self.assertRaises(RuntimeError, DataManager, self.path, self.empty_file, self.file)
        self.assertRaises(RuntimeError, DataManager, self.path, self.file, self.empty_file)

    def test_bad_keyword_arguments(self):
        self.assertRaises((TypeError, FileNotFoundError), DataManager, self.path, self.file, self.file,
                          triggered_test_file=0)
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          train_data_transform=object())
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          test_data_transform=object())
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          data_loader=0)
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          train_label_transform='string')
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          test_label_transform='string')
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          shuffle_train='string')
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          shuffle_clean_test=2)
        self.assertRaises(TypeError, DataManager, self.path, self.file, self.file,
                          shuffle_triggered_test=object())


if __name__ == "__main__":
    unittest.main()
