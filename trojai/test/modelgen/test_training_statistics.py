import unittest
import tempfile
import pandas as pd
import os
import numpy as np

import trojai.modelgen.training_statistics as tpm_ts


class TestTrainingStatistics(unittest.TestCase):
    def test_batch_statistics(self):
        batch_num = 1
        batch_train_acc = 50
        batch_train_loss = -4
        batch_val_acc = 75
        batch_val_loss = -3
        batch_statistics = tpm_ts.BatchStatistics(batch_num,
                                                  batch_train_acc, batch_train_loss,
                                                  batch_val_acc, batch_val_loss)
        self.assertEqual(batch_statistics.get_batch_num(), batch_num)
        self.assertEqual(batch_statistics.get_batch_train_acc(), batch_train_acc)
        self.assertEqual(batch_statistics.get_batch_train_loss(), batch_train_loss)
        self.assertEqual(batch_statistics.get_batch_validation_acc(), batch_val_acc)
        self.assertEqual(batch_statistics.get_batch_validation_loss(), batch_val_loss)

        self.assertRaises(ValueError, batch_statistics.set_batch_train_acc, 150)
        self.assertRaises(ValueError, batch_statistics.set_batch_train_acc, -50)
        self.assertRaises(ValueError, batch_statistics.set_batch_validation_acc, 150)
        self.assertRaises(ValueError, batch_statistics.set_batch_validation_acc, -50)

    def test_epoch_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 1, 1, 1)
        batch2 = tpm_ts.BatchStatistics(2, 2, 2, 2, 2)
        batch3 = tpm_ts.BatchStatistics(3, 3, 3, 3, 3)
        batch4 = tpm_ts.BatchStatistics(4, 4, 4, 4, 4)

        epoch_stats = tpm_ts.EpochStatistics(1)
        epoch_stats.add_batch(batch1)
        epoch_stats.add_batch(batch2)
        epoch_stats.add_batch([batch3, batch4])

        batch_stats_list = epoch_stats.get_batch_stats()
        for batch_idx, batch_stats in enumerate(batch_stats_list):
            expected_val = batch_idx + 1
            self.assertEqual(batch_stats.get_batch_num(), expected_val)
            self.assertEqual(batch_stats.get_batch_train_acc(), expected_val)
            self.assertEqual(batch_stats.get_batch_train_loss(), expected_val)
            self.assertEqual(batch_stats.get_batch_validation_acc(), expected_val)
            self.assertEqual(batch_stats.get_batch_validation_loss(), expected_val)
        self.assertEqual(epoch_stats.get_epoch_num(), 1)

    def test_training_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 1, 1, 1)
        batch2 = tpm_ts.BatchStatistics(2, 2, 2, 2, 2)
        batch3 = tpm_ts.BatchStatistics(3, 3, 3, 3, 3)
        batch4 = tpm_ts.BatchStatistics(4, 4, 4, 4, 4)
        batch5 = tpm_ts.BatchStatistics(5, 5, 5, 5, 5)
        batch6 = tpm_ts.BatchStatistics(6, 6, 6, 6, 6)

        epoch1_stats = tpm_ts.EpochStatistics(1)
        epoch2_stats = tpm_ts.EpochStatistics(2)
        epoch3_stats = tpm_ts.EpochStatistics(3)
        epoch1_stats.add_batch([batch1, batch2])
        epoch2_stats.add_batch([batch3, batch4])
        epoch3_stats.add_batch([batch5, batch6])

        training_stats = tpm_ts.TrainingRunStatistics()
        training_stats.add_epoch(epoch1_stats)
        training_stats.add_epoch([epoch2_stats, epoch3_stats])
        training_stats.set_final_train_acc(1)
        training_stats.set_final_train_loss(1)
        training_stats.set_final_val_acc(1)
        training_stats.set_final_val_loss(1)
        training_stats.set_final_clean_data_test_acc(1)
        training_stats.set_final_triggered_data_test_acc(1)

        summary_dict = training_stats.get_summary()
        self.assertEqual(summary_dict['final_train_acc'], 1)
        self.assertEqual(summary_dict['final_train_loss'], 1)
        self.assertEqual(summary_dict['final_val_acc'], 1)
        self.assertEqual(summary_dict['final_val_loss'], 1)
        self.assertEqual(summary_dict['final_clean_data_test_acc'], 1)
        self.assertEqual(summary_dict['final_triggered_data_test_acc'], 1)

        self.assertRaises(ValueError, training_stats.set_final_train_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_train_acc, -50)
        self.assertRaises(ValueError, training_stats.set_final_val_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_val_acc, -50)
        self.assertRaises(ValueError, training_stats.set_final_clean_data_test_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_clean_data_test_acc, -50)
        self.assertRaises(ValueError, training_stats.set_final_triggered_data_test_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_triggered_data_test_acc, -50)

        # ensure data is maintained over epochs
        epoch_stats = training_stats.get_epochs_stats()
        batch_idx = 1
        for epoch_num, epoch in enumerate(epoch_stats):
            actual_epoch_num = epoch_num + 1
            self.assertEqual(epoch.get_epoch_num(), actual_epoch_num)
            batch_stats_list = epoch.get_batch_stats()
            for batch_stats in batch_stats_list:
                expected_val = batch_idx
                self.assertEqual(batch_stats.get_batch_num(), expected_val)
                self.assertEqual(batch_stats.get_batch_train_acc(), expected_val)
                self.assertEqual(batch_stats.get_batch_train_loss(), expected_val)
                self.assertEqual(batch_stats.get_batch_validation_acc(), expected_val)
                self.assertEqual(batch_stats.get_batch_validation_loss(), expected_val)

                batch_idx += 1

    def test_autopopulate_training_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 1, 1, 1)
        batch2 = tpm_ts.BatchStatistics(2, 2, 2, 2, 2)
        batch3 = tpm_ts.BatchStatistics(3, 3, 3, 3, 3)
        batch4 = tpm_ts.BatchStatistics(4, 4, 4, 4, 4)
        batch5 = tpm_ts.BatchStatistics(5, 5, 5, 5, 5)
        batch6 = tpm_ts.BatchStatistics(6, 6, 6, 6, 6)

        epoch1_stats = tpm_ts.EpochStatistics(1)
        epoch2_stats = tpm_ts.EpochStatistics(2)
        epoch3_stats = tpm_ts.EpochStatistics(3)
        epoch1_stats.add_batch([batch1, batch2])
        epoch2_stats.add_batch([batch3, batch4])
        epoch3_stats.add_batch([batch5, batch6])

        training_stats = tpm_ts.TrainingRunStatistics()
        training_stats.add_epoch(epoch1_stats)
        training_stats.add_epoch([epoch2_stats, epoch3_stats])
        training_stats.add_num_epochs_trained(1)
        training_stats.autopopulate_final_summary_stats()
        training_stats.set_final_clean_data_test_acc(1)
        training_stats.set_final_triggered_data_test_acc(1)

        summary_dict = training_stats.get_summary()
        self.assertEqual(summary_dict['final_train_acc'], 6)
        self.assertEqual(summary_dict['final_train_loss'], 6)
        self.assertEqual(summary_dict['final_val_acc'], 6)
        self.assertEqual(summary_dict['final_val_loss'], 6)
        self.assertEqual(summary_dict['final_clean_data_test_acc'], 1)
        self.assertEqual(summary_dict['final_triggered_data_test_acc'], 1)

    def test_save_detailed_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 2, 3, 4)
        batch2 = tpm_ts.BatchStatistics(2, 5, 6, 7, 8)
        batch3 = tpm_ts.BatchStatistics(1, 9, 10, 11, 12)
        batch4 = tpm_ts.BatchStatistics(2, 13, 14, 15, 16)
        batch5 = tpm_ts.BatchStatistics(1, 17, 18, 19, 20)
        batch6 = tpm_ts.BatchStatistics(2, 21, 22, 23, 24)

        epoch1_stats = tpm_ts.EpochStatistics(1)
        epoch2_stats = tpm_ts.EpochStatistics(2)
        epoch3_stats = tpm_ts.EpochStatistics(3)
        epoch1_stats.add_batch([batch1, batch2])
        epoch2_stats.add_batch([batch3, batch4])
        epoch3_stats.add_batch([batch5, batch6])

        training_stats = tpm_ts.TrainingRunStatistics()
        training_stats.add_epoch(epoch1_stats)
        training_stats.add_epoch([epoch2_stats, epoch3_stats])

        output_file = tempfile.NamedTemporaryFile(delete=False)
        fname = output_file.name
        output_file.close()
        training_stats.save_detailed_stats_to_disk(fname)
        # read in the file w/ pandas and ensure data consistency
        df = pd.read_csv(fname)
        self.assertTrue(np.array_equal(df['epoch_number'].values, np.asarray([1, 1, 2, 2, 3, 3])))
        self.assertTrue(np.array_equal(df['batch_num'].values, np.asarray([0, 1, 0, 1, 0, 1])))
        self.assertTrue(np.array_equal(df['train_accuracy'].values, np.asarray([1, 5, 9, 13, 17, 21])))
        self.assertTrue(np.array_equal(df['train_loss'].values, np.asarray([2, 6, 10, 14, 18, 22])))
        self.assertTrue(np.array_equal(df['val_acc'].values, np.asarray([3, 7, 11, 15, 19, 23])))
        self.assertTrue(np.array_equal(df['val_loss'].values, np.asarray([4, 8, 12, 16, 20, 24])))

        # delete file
        os.unlink(fname)


if __name__ == '__main__':
    unittest.main()
