import unittest
import tempfile
import pandas as pd
import numpy as np

import trojai.modelgen.training_statistics as tpm_ts


class TestTrainingStatistics(unittest.TestCase):
    def test_batch_statistics(self):
        batch_num = 1
        batch_train_acc = 50
        batch_train_loss = -4
        batch_statistics = tpm_ts.BatchStatistics(batch_num,
                                                  batch_train_acc, batch_train_loss)
        self.assertEqual(batch_statistics.get_batch_num(), batch_num)
        self.assertEqual(batch_statistics.get_batch_train_acc(), batch_train_acc)
        self.assertEqual(batch_statistics.get_batch_train_loss(), batch_train_loss)

        self.assertRaises(ValueError, batch_statistics.set_batch_train_acc, 150)
        self.assertRaises(ValueError, batch_statistics.set_batch_train_acc, -50)

    def test_epoch_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 1)
        batch2 = tpm_ts.BatchStatistics(2, 2, 2)
        batch3 = tpm_ts.BatchStatistics(3, 3, 3)
        batch4 = tpm_ts.BatchStatistics(4, 4, 4)

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
        self.assertEqual(epoch_stats.get_epoch_num(), 1)

    def test_training_statistics(self):
        batch1 = tpm_ts.BatchStatistics(1, 1, 1)
        batch2 = tpm_ts.BatchStatistics(2, 2, 2)
        batch3 = tpm_ts.BatchStatistics(3, 3, 3)
        batch4 = tpm_ts.BatchStatistics(4, 4, 4)
        batch5 = tpm_ts.BatchStatistics(5, 5, 5)
        batch6 = tpm_ts.BatchStatistics(6, 6, 6)

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
        training_stats.set_final_val_combined_acc(1)
        training_stats.set_final_val_combined_loss(1)
        training_stats.set_final_clean_data_test_acc(1)
        training_stats.set_final_triggered_data_test_acc(1)

        summary_dict = training_stats.get_summary()
        self.assertEqual(summary_dict['final_train_acc'], 1)
        self.assertEqual(summary_dict['final_train_loss'], 1)
        self.assertEqual(summary_dict['final_combined_val_acc'], 1)
        self.assertEqual(summary_dict['final_combined_val_loss'], 1)
        self.assertEqual(summary_dict['final_clean_data_test_acc'], 1)
        self.assertEqual(summary_dict['final_triggered_data_test_acc'], 1)

        self.assertRaises(ValueError, training_stats.set_final_train_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_train_acc, -50)
        self.assertRaises(ValueError, training_stats.set_final_val_combined_acc, 150)
        self.assertRaises(ValueError, training_stats.set_final_val_combined_acc, -50)
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

                batch_idx += 1

    def test_autopopulate_training_statistics(self):
        training_stats = tpm_ts.TrainingRunStatistics()
        epoch1_stats = tpm_ts.EpochStatistics(1, tpm_ts.EpochTrainStatistics(.5, .5),
                                              tpm_ts.EpochValidationStatistics(.5, .5, None, None))
        epoch2_stats = tpm_ts.EpochStatistics(2, tpm_ts.EpochTrainStatistics(.6, .6),
                                              tpm_ts.EpochValidationStatistics(.6, .6, None, None))
        epoch3_stats = tpm_ts.EpochStatistics(3, tpm_ts.EpochTrainStatistics(.7, .8),
                                              tpm_ts.EpochValidationStatistics(.9, 1.1, None, None))
        training_stats.add_epoch(epoch1_stats)
        training_stats.add_epoch([epoch2_stats, epoch3_stats])
        training_stats.add_num_epochs_trained(3)

        training_stats.autopopulate_final_summary_stats()
        training_stats.set_final_clean_data_test_acc(1)
        training_stats.set_final_triggered_data_test_acc(1)

        summary_dict = training_stats.get_summary()
        self.assertEqual(summary_dict['final_train_acc'], .7)
        self.assertEqual(summary_dict['final_train_loss'], .8)
        self.assertEqual(summary_dict['final_combined_val_acc'], .9)
        self.assertEqual(summary_dict['final_combined_val_loss'], 1.1)
        self.assertEqual(summary_dict['final_clean_data_test_acc'], 1)
        self.assertEqual(summary_dict['final_triggered_data_test_acc'], 1)

    def test_save_detailed_statistics(self):
        epoch1_stats = tpm_ts.EpochStatistics(1, tpm_ts.EpochTrainStatistics(.5, 1.5),
                                              tpm_ts.EpochValidationStatistics(2.5, 3.5, None, None))
        epoch2_stats = tpm_ts.EpochStatistics(2, tpm_ts.EpochTrainStatistics(.6, 1.6),
                                              tpm_ts.EpochValidationStatistics(2.6, 3.6, None, None))
        epoch3_stats = tpm_ts.EpochStatistics(3, tpm_ts.EpochTrainStatistics(.7, 1.7),
                                              tpm_ts.EpochValidationStatistics(2.7, 3.7, None, None))

        training_stats = tpm_ts.TrainingRunStatistics()
        training_stats.add_epoch(epoch1_stats)
        training_stats.add_epoch([epoch2_stats, epoch3_stats])

        with tempfile.NamedTemporaryFile() as output_file:
            fname = output_file.name
            training_stats.save_detailed_stats_to_disk(fname)
            # read in the file w/ pandas and ensure data consistency
            df = pd.read_csv(fname)
            self.assertTrue(np.array_equal(df['epoch_number'].values, np.asarray([1, 2, 3])))
            self.assertTrue(np.array_equal(df['train_acc'].values, np.asarray([.5, .6, .7])))
            self.assertTrue(np.array_equal(df['train_loss'].values, np.asarray([1.5, 1.6, 1.7])))
            self.assertTrue(np.array_equal(df['combined_val_acc'].values, np.asarray([2.5, 2.6, 2.7])))
            self.assertTrue(np.array_equal(df['combined_val_loss'].values, np.asarray([3.5, 3.6, 3.7])))


if __name__ == '__main__':
    unittest.main()
