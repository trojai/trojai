import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from trojai.modelgen.default_optimizer import DefaultOptimizer, _eval_acc, train_val_dataset_split
from trojai.modelgen.config import DefaultOptimizerConfig, TrainingConfig, EarlyStoppingConfig
from trojai.modelgen.training_statistics import BatchStatistics

"""
Contains unittests related to 
"""
SEED = 1234


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.rso = np.random.RandomState(SEED)

    def tearDown(self):
        pass

    def test_accuracy(self):
        cfg = DefaultOptimizerConfig()
        optimizer_obj = DefaultOptimizer(cfg)
        batch_size = cfg.training_cfg.batch_size
        num_outputs = 5

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat/row_sum[:,None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size*(1-target_accuracy))
        num_indices_unmodified = batch_size-num_indices_to_modify
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)
        expected_accuracy = float(num_indices_unmodified)/float(batch_size) * 100

        for ii in indices_to_modify:
            true_output[ii] = true_output[ii]+1

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=0, n_correct=0)
        self.assertAlmostEqual(actual_acc, expected_accuracy)

    def test_running_accuracy(self):
        batch_size = 32
        num_outputs = 5

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat/row_sum[:, None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size*(1-target_accuracy))
        num_indices_unmodified = batch_size-num_indices_to_modify
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)

        for ii in indices_to_modify:
            true_output[ii] = true_output[ii]+1

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        n_total_prev = 64
        n_correct_prev = 50
        expected_accuracy = (num_indices_unmodified+n_correct_prev)/(batch_size+n_total_prev) * 100
        expected_n_total = n_total_prev+batch_size
        expected_n_correct = n_correct_prev + num_indices_unmodified

        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=n_total_prev, n_correct=n_correct_prev)
        self.assertAlmostEqual(actual_acc, expected_accuracy)
        self.assertEqual(expected_n_total, n_total)
        self.assertEqual(expected_n_correct, n_correct)

    def test_eval_binary_accuracy(self):
        batch_size = 32
        num_outputs = 2

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat/row_sum[:, None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size*(1-target_accuracy))
        num_indices_unmodified = batch_size-num_indices_to_modify
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)

        for ii in indices_to_modify:
            true_output[ii] += 1

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        n_total_prev = 64
        n_correct_prev = 50
        expected_accuracy = (num_indices_unmodified+n_correct_prev)/(batch_size+n_total_prev) * 100
        expected_n_total = n_total_prev+batch_size
        expected_n_correct = n_correct_prev + num_indices_unmodified

        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=n_total_prev, n_correct=n_correct_prev)
        self.assertAlmostEqual(actual_acc, expected_accuracy)
        self.assertEqual(expected_n_total, n_total)
        self.assertEqual(expected_n_correct, n_correct)

    def test_eval_binary_one_output_accuracy(self):
        batch_size = 32
        num_outputs = 1

        true_output = self.rso.rand(batch_size, num_outputs)*5-10  # test output between -5 and 5
        true_output_binary = np.expand_dims(np.asarray([0 if x < 0 else 1 for x in true_output], dtype=np.int), axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        network_output = true_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size*(1-target_accuracy))
        num_indices_unmodified = batch_size-num_indices_to_modify
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)

        for ii in indices_to_modify:
            # flip pos to neg, neg to pos
            if network_output[ii][0] >= 0:
                network_output[ii][0] = network_output[ii][0] - 10
            else:
                network_output[ii][0] = network_output[ii][0] + 10

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output_binary, dtype=torch.long)

        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=0, n_correct=0)
        expected_acc = float(batch_size-num_indices_to_modify)/batch_size*100
        expected_n_total = 32
        expected_n_correct = num_indices_unmodified
        self.assertAlmostEqual(actual_acc, expected_acc)
        self.assertEqual(n_total, expected_n_total)
        self.assertEqual(n_correct, expected_n_correct)

        # now compute the accuracy
        n_total_prev = 64
        n_correct_prev = 50
        expected_accuracy = (num_indices_unmodified+n_correct_prev)/(batch_size+n_total_prev) * 100
        expected_n_total = n_total_prev+batch_size
        expected_n_correct = n_correct_prev + num_indices_unmodified

        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=n_total_prev, n_correct=n_correct_prev)
        self.assertAlmostEqual(actual_acc, expected_accuracy)
        self.assertEqual(expected_n_total, n_total)
        self.assertEqual(expected_n_correct, n_correct)

    def test_train_val_split(self):
        t1 = torch.Tensor(np.arange(10))
        dataset = torch.utils.data.TensorDataset(t1)
        split_amt = 0.2
        train_dataset, val_dataset = train_val_dataset_split(dataset, split_amt)
        self.assertEqual(len(train_dataset), int(len(t1)*(1-split_amt)))
        self.assertEqual(len(val_dataset), int(len(t1)*split_amt))

    def test_train_val_split2(self):
        t1 = torch.Tensor(np.arange(10))
        dataset = torch.utils.data.TensorDataset(t1)
        split_amt = 0.0
        train_dataset, val_dataset = train_val_dataset_split(dataset, split_amt)
        self.assertEqual(len(train_dataset), int(len(t1)*(1-split_amt)))
        self.assertEqual(len(val_dataset), int(len(t1)*split_amt))

    def test_str(self):
        training_cfg = TrainingConfig(device='cpu')
        cfg = DefaultOptimizerConfig(training_cfg)
        optimizer_obj = DefaultOptimizer(cfg)
        optimizer_string = str(optimizer_obj)
        correct_string = "{'batch_size':32, 'num_epochs':10, " \
                         "'device':'cpu', 'lr':1.00000e-04, 'loss_function':'cross_entropy_loss', 'optimizer':'adam'}"
        self.assertEqual(optimizer_string, correct_string)

    def test_early_stopping1(self):
        """
        The purpose of this test is to ensure that the early stopping is activated.  The test works by Mocking the
        train_epoch function.  After 4 epochs, the train_epoch function returns BatchStatistics with val_acc less
        than the threshold required, such that early-stopping occurs on the 9th epoch.
        """
        optimizer = DefaultOptimizer()
        optimizer.optimizer_cfg.training_cfg.epochs = 10

        optimizer.optimizer_cfg.training_cfg.early_stopping = EarlyStoppingConfig()
        model = Mock(spec=nn.Module)
        model.parameters = Mock()
        dataset = Mock(spec=torch.utils.data.Dataset)

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
             patch('trojai.modelgen.default_optimizer.train_val_dataset_split', return_value=([], [])) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, compute_batch_stats,
                                        progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                batch_num_no_op = 999
                batch_train_acc_noop = 1
                batch_train_loss_noop = 1
                batch_val_loss_noop = 1
                if epoch < 4:
                    batch_val_acc = epoch # we just set the accuracy to an integer that increases over every epoch.
                                          # This prevents the early-stopping code from being activated,
                                          # since the accuracy is changing by a drastic amount every epoch
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]
                else:
                    batch_val_acc = optimizer.optimizer_cfg.training_cfg.early_stopping.val_acc_eps
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]
            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # 4 is when the early-stopping stats are modified
            # 5 is the default # of epochs to monitor for early stopping
            # see: EarlyStoppingConfig() for further details
            self.assertEqual(num_epochs_trained, 4+5)

    def test_early_stopping2(self):
        """
        The purpose of this test is to ensure that the early stopping is not activated, when the configuration for
        EarlyStopping is set to None.  Even though we modify the val_acc as before, after epoch 4, EarlyStopping is
        not configured and as a result, we train for all 10 epochs
        """
        optimizer_cfg = DefaultOptimizerConfig()
        optimizer_cfg.training_cfg.device = torch.device('cuda')  # trick the device so that no warnings are triggered
                                                                  # upon instantiation of the DefaultOptimizer
        optimizer = DefaultOptimizer(optimizer_cfg)
        optimizer.device = Mock()
        optimizer.optimizer_cfg.training_cfg.epochs = 10

        optimizer.optimizer_cfg.training_cfg.early_stopping = None
        model = Mock(spec=nn.Module)
        model.parameters = Mock()
        dataset = Mock(spec=torch.utils.data.Dataset)

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
            patch('trojai.modelgen.default_optimizer.train_val_dataset_split',
                  return_value=([], [])) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, compute_batch_stats,
                                        progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                batch_num_no_op = 999
                batch_train_acc_noop = 1
                batch_train_loss_noop = 1
                batch_val_loss_noop = 1
                if epoch < 4:
                    batch_val_acc = epoch  # we just set the accuracy to an integer that increases over every epoch.
                    # This prevents the early-stopping code from being activated,
                    # since the accuracy is changing by a drastic amount every epoch
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]
                else:
                    batch_val_acc = float(1e-8)
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]

            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # the early stopping should *not* have been run, b/c we set it to None, so we should
            # have trained for the full 10 epochs
            self.assertEqual(num_epochs_trained, optimizer.optimizer_cfg.training_cfg.epochs)

    def test_early_stopping3(self):
        """
        The purpose of this test is to ensure that the early stopping is not activated - for the case where the
        EarlyStopping criterion of the number of epochs over which the validation accuracy must be lower than the
        threshold is not reached before end of training.
        """
        optimizer_cfg = DefaultOptimizerConfig()
        optimizer_cfg.training_cfg.device = torch.device('cuda')  # trick the device so that no warnings are triggered
        # upon instantiation of the DefaultOptimizer
        optimizer = DefaultOptimizer(optimizer_cfg)
        optimizer.optimizer_cfg.training_cfg.epochs = 10

        optimizer.optimizer_cfg.training_cfg.early_stopping = EarlyStoppingConfig()
        model = Mock(spec=nn.Module)
        model.parameters = Mock()
        dataset = Mock(spec=torch.utils.data.Dataset)

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
            patch('trojai.modelgen.default_optimizer.train_val_dataset_split',
                  return_value=([], [])) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, compute_batch_stats,
                                        progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                batch_num_no_op = 999
                batch_train_acc_noop = 1
                batch_train_loss_noop = 1
                batch_val_loss_noop = 1
                if epoch < 9:
                    batch_val_acc = epoch  # we just set the accuracy to an integer that increases over every epoch.
                                           # This prevents the early-stopping code from being activated,
                                           # since the accuracy is changing by a drastic amount every epoch
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]
                else:
                    batch_val_acc = optimizer.optimizer_cfg.training_cfg.early_stopping.val_acc_eps
                    return [BatchStatistics(batch_num_no_op, batch_train_acc_noop, batch_train_loss_noop,
                                            batch_val_acc, batch_val_loss_noop)]

            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # now put in the test condition
            # the test condition here that no warnings are triggered upon instantiation.
            # this means that early stopping code should not have been run, and we should
            # have trained for the full 10 epochs
            self.assertEqual(num_epochs_trained, optimizer.optimizer_cfg.training_cfg.epochs)

    # TODO: add mock tests on saving best model


if __name__ == '__main__':
    unittest.main()
