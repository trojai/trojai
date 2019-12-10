import unittest
import tempfile
import os
import random

import pandas as pd

from trojai.modelgen.datasets import CSVDataset


class TestDataManager(unittest.TestCase):
    tmp_dir = tempfile.TemporaryDirectory()
    data_path = tmp_dir.name
    data_filename = 'test_file.csv'

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        test_data = pd.DataFrame.from_dict({'A': {0: 0, 1: 2}, 'B': {0: 1, 1: 3}, 'train_label': {0: 0, 1: 1}})
        test_data.to_csv(os.path.join(self.data_path, self.data_filename), index=False)
        random.seed(1234)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_datasets(self):
        # Note that inputs for these objects should have been validated previously by DataManager and its
        # config object
        data_path = self.data_path
        data_filename = self.data_filename
        csv_dataset = CSVDataset(data_path, data_filename, data_loader=lambda x: x + 1, data_transform=lambda x: x ** 2,
                                 label_transform=lambda x: str(x))
        correct_dataframe = pd.DataFrame.from_dict({'A': {0: 0, 1: 2}, 'B': {0: 1, 1: 3}, 'train_label': {0: 0, 1: 1}})
        self.assertTrue(csv_dataset.data_df.equals(correct_dataframe))
        self.assertEqual(csv_dataset.data_loader(1), 2)
        self.assertEqual(csv_dataset.data_transform(2), 4)
        self.assertEqual(csv_dataset.label_transform(1), '1')

        csv_dataset = CSVDataset(data_path, data_filename, shuffle=True, random_state=1234, data_loader=lambda x: x,
                                 data_transform=lambda x: 2*x, label_transform=lambda x: 50)
        self.assertTrue(csv_dataset.data_df.equals(correct_dataframe))
        self.assertEqual(csv_dataset.data_loader(1), 1)
        self.assertEqual(csv_dataset.data_transform(3), 6)
        self.assertEqual(csv_dataset.label_transform(1), 50)

        # changed rows
        correct_dataframe = pd.DataFrame.from_dict({'A': {0: 2, 1: 0}, 'B': {0: 3, 1: 1}, 'train_label': {0: 1,
                                                                                                          1: 0}})
        csv_dataset = CSVDataset(data_path, data_filename, shuffle=True, random_state=123, data_loader=lambda x: -x,
                                 data_transform=lambda x: x / 2, label_transform=lambda x: 50)
        self.assertTrue(csv_dataset.data_df.equals(correct_dataframe))
        self.assertEqual(csv_dataset.data_loader(1), -1)
        self.assertEqual(csv_dataset.data_transform(6), 3)
        self.assertEqual(csv_dataset.label_transform(1), 50)

        # should we do this?
        # Todo: test default data loader(s) as desired, e.g. 'image' -> loads correct image from file location


if __name__ == "__main__":
    unittest.main()
