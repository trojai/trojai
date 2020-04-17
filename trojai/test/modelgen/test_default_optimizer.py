import unittest
from unittest.mock import Mock, patch
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from trojai.modelgen.default_optimizer import DefaultOptimizer, _eval_acc, train_val_dataset_split
from trojai.modelgen.config import DefaultOptimizerConfig, TrainingConfig, EarlyStoppingConfig
from trojai.modelgen.training_statistics import EpochTrainStatistics, EpochValidationStatistics

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
        batch_size = cfg.training_cfg.batch_size
        num_outputs = 5

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat / row_sum[:, None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size * (1 - target_accuracy))
        num_indices_unmodified = batch_size - num_indices_to_modify
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)
        expected_accuracy = float(num_indices_unmodified) / float(batch_size) * 100

        for ii in indices_to_modify:
            true_output[ii] = true_output[ii] + 1

        expected_balanced_acc = balanced_accuracy_score(true_output, fake_network_output.argmax(axis=1))*100

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=None, n_correct=None)
        self.assertAlmostEqual(actual_acc, expected_balanced_acc)

    def test_running_accuracy(self):
        batch_size = 32
        num_outputs = 5

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat / row_sum[:, None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size * (1 - target_accuracy))
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)

        for ii in indices_to_modify:
            true_output[ii] = (true_output[ii] + 1) % num_outputs

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        n_total_prev = defaultdict(int, {0: 13, 1: 13, 2: 13, 3: 13, 4: 12})
        n_correct_prev = defaultdict(int, {0: 11, 1: 11, 2: 11, 3: 11, 4: 11})

        n_total_expected = n_total_prev.copy()
        n_correct_expected = n_correct_prev.copy()
        for to in true_output:
            n_total_expected[to] += 1
        # compute how many the network got right, and update the necessary output
        indices_not_modified = set(range(batch_size)).symmetric_difference(set(indices_to_modify))
        for ii in indices_not_modified:
            n_correct_expected[network_output[ii]] += 1

        # update the true & fake_network_output to aggregate both the previous call to _eval_acc
        # and the current call to _eval_acc
        true_output_prev_and_cur = list(true_output)
        network_output_prev_and_cur = list(network_output)
        for k, v, in n_total_prev.items():
            true_output_prev_and_cur.extend([k]*v)
        # simulate network outputs to keep the correct & total counts according to the previously defined dictionaries
        for k, v in n_correct_prev.items():
            num_correct = v
            num_incorrect = n_total_prev[k]-num_correct
            network_output_prev_and_cur.extend([k] * num_correct)
            network_output_prev_and_cur.extend([((k+1) % num_outputs)] * num_incorrect)
        expected_balanced_acc = balanced_accuracy_score(true_output_prev_and_cur, network_output_prev_and_cur)*100

        actual_acc, n_total_actual, n_correct_actual = \
            _eval_acc(network_output_pt, true_output_pt, n_total=n_total_prev, n_correct=n_correct_prev)

        self.assertAlmostEqual(actual_acc, expected_balanced_acc)
        self.assertEqual(n_total_expected, n_total_actual)
        self.assertEqual(n_correct_expected, n_correct_actual)

    def test_eval_binary_accuracy(self):
        batch_size = 32
        num_outputs = 2

        random_mat = self.rso.rand(batch_size, num_outputs)
        row_sum = random_mat.sum(axis=1)

        # normalize the random_mat such that every row adds up to 1
        # broadcast so we can divide every element in matrix by the row's sum
        fake_network_output = random_mat / row_sum[:, None]
        network_output = np.argmax(fake_network_output, axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        true_output = network_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size * (1 - target_accuracy))
        indices_to_modify = self.rso.choice(range(batch_size), num_indices_to_modify, replace=False)

        for ii in indices_to_modify:
            true_output[ii] = (true_output[ii] + 1) % num_outputs

        # convert datatypes to what is expected during operation
        network_output_pt = torch.tensor(fake_network_output, dtype=torch.float)
        true_output_pt = torch.tensor(true_output, dtype=torch.long)

        # now compute the accuracy
        n_total_prev = defaultdict(int, {0: 40, 1: 24})
        n_correct_prev = defaultdict(int, {0: 11, 1: 11})

        n_total_expected = n_total_prev.copy()
        n_correct_expected = n_correct_prev.copy()
        for to in true_output:
            n_total_expected[to] += 1
        # compute how many the network got right, and update the necessary output
        indices_not_modified = set(range(batch_size)).symmetric_difference(set(indices_to_modify))
        for ii in indices_not_modified:
            n_correct_expected[network_output[ii]] += 1

        # update the true & fake_network_output to aggregate both the previous call to _eval_acc
        # and the current call to _eval_acc
        true_output_prev_and_cur = list(true_output)
        network_output_prev_and_cur = list(network_output)
        for k, v, in n_total_prev.items():
            true_output_prev_and_cur.extend([k] * v)
        # simulate network outputs to keep the correct & total counts according to the previously defined dictionaries
        for k, v in n_correct_prev.items():
            num_correct = v
            num_incorrect = n_total_prev[k] - num_correct
            network_output_prev_and_cur.extend([k] * num_correct)
            network_output_prev_and_cur.extend([((k + 1) % num_outputs)] * num_incorrect)
        expected_balanced_acc = balanced_accuracy_score(true_output_prev_and_cur, network_output_prev_and_cur) * 100

        actual_acc, n_total, n_correct = \
            _eval_acc(network_output_pt, true_output_pt, n_total=n_total_prev, n_correct=n_correct_prev)
        self.assertAlmostEqual(actual_acc, expected_balanced_acc)
        self.assertEqual(n_total_expected, n_total)
        self.assertEqual(n_correct_expected, n_correct)

    def test_eval_binary_one_output_accuracy(self):
        batch_size = 32
        num_outputs = 1

        true_output = (self.rso.rand(batch_size, num_outputs) * 4) - 2
        true_output_binary = np.expand_dims(np.asarray([0 if x < 0 else 1 for x in true_output],
                                                       dtype=np.int), axis=1)

        # now, modify a subset of the netowrk output and make that the "real" output
        network_output = true_output.copy()
        target_accuracy = 0.8
        num_indices_to_modify = int(batch_size * (1 - target_accuracy))
        num_indices_unmodified = batch_size - num_indices_to_modify
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
            _eval_acc(network_output_pt, true_output_pt, n_total=None, n_correct=None)

        expected_n_total = defaultdict(int)
        sigmoid_fn = lambda x: 1./(1. + np.exp(-x))
        for ii in range(len(true_output_binary)):
            to = true_output_binary[ii][0]
            expected_n_total[to] += 1
        expected_n_correct = defaultdict(int)
        for ii in range(batch_size):
            expected_n_correct[true_output_binary[ii][0]] += int(np.round(sigmoid_fn(network_output[ii][0])) ==
                                                                 true_output_binary[ii][0])
        expected_acc = balanced_accuracy_score(true_output_binary, np.round(sigmoid_fn(network_output)))*100
        self.assertAlmostEqual(actual_acc, expected_acc)
        self.assertEqual(n_total, expected_n_total)
        self.assertEqual(n_correct, expected_n_correct)

        # now, test running accuracy on one neuron output
        additional_n_total = 20
        additional_n_correct = 15
        new_network_output = np.zeros((additional_n_total, 1))
        new_true_output_binary = np.zeros((additional_n_total, 1))
        new_expected_n_total = expected_n_total.copy()
        new_expected_n_correct = expected_n_correct.copy()
        for ii in range(additional_n_total):
            bin_val = self.rso.randint(2)
            new_true_output_binary[ii][0] = bin_val
            new_expected_n_total[bin_val] += 1
            if ii < additional_n_correct:
                new_network_output[ii][0] = 10 if bin_val == 1 else -10
                new_expected_n_correct[bin_val] += 1
            else:
                new_network_output[ii][0] = -10 if bin_val == 1 else 10

        new_network_output_pt = torch.tensor(new_network_output, dtype=torch.float)
        new_true_output_pt = torch.tensor(new_true_output_binary, dtype=torch.long)

        # concatenate the network output and true output to format for the _eval_acc function
        new_network_output_concat = np.vstack((network_output, new_network_output))
        new_true_output_concat = np.vstack((true_output_binary, new_true_output_binary))
        new_expected_acc = balanced_accuracy_score(new_true_output_concat,
                                                   np.round(sigmoid_fn(new_network_output_concat)))*100

        actual_acc, n_total, n_correct = \
            _eval_acc(new_network_output_pt, new_true_output_pt, n_total=expected_n_total, n_correct=expected_n_correct)
        self.assertAlmostEqual(actual_acc, new_expected_acc)
        self.assertEqual(new_expected_n_total, n_total)
        self.assertEqual(new_expected_n_correct, n_correct)

    def test_train_val_split(self):
        t1 = torch.Tensor(np.arange(10))
        dataset = torch.utils.data.TensorDataset(t1)
        split_amt = 0.2

        def val_data_xform(x): return x

        def val_label_xform(y): return y ** 2

        train_dataset, val_dataset = train_val_dataset_split(dataset, split_amt, val_data_xform, val_label_xform)
        self.assertEqual(len(train_dataset), int(len(t1) * (1 - split_amt)))
        self.assertEqual(len(val_dataset), int(len(t1) * split_amt))
        self.assertEqual(val_dataset.data_transform, val_data_xform)
        self.assertEqual(val_dataset.label_transform, val_label_xform)

    def test_train_val_split2(self):
        t1 = torch.Tensor(np.arange(10))
        dataset = torch.utils.data.TensorDataset(t1)
        split_amt = 0.0

        def val_data_xform(x): return x

        def val_label_xform(y): return y ** 2

        train_dataset, val_dataset = train_val_dataset_split(dataset, split_amt, val_data_xform, val_label_xform)
        self.assertEqual(len(train_dataset), int(len(t1) * (1 - split_amt)))
        self.assertEqual(len(val_dataset), int(len(t1) * split_amt))
        self.assertEqual(val_dataset.data_transform, val_data_xform)
        self.assertEqual(val_dataset.label_transform, val_label_xform)

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
        train_epoch function.  After 2 epochs, the train_epoch function returns BatchStatistics with val_less less
        than the threshold required, such that early-stopping occurs on the 8th epoch.
        """
        optimizer = DefaultOptimizer()
        optimizer.optimizer_cfg.training_cfg.epochs = 10

        optimizer.optimizer_cfg.training_cfg.early_stopping = EarlyStoppingConfig()
        eps = optimizer.optimizer_cfg.training_cfg.early_stopping.val_loss_eps
        model = Mock(spec=nn.Module)
        model.parameters = Mock()
        dataset = Mock(spec=torch.utils.data.BatchSampler)
        dataset.__len__ = Mock(return_value=2)  # otherwise the pytorch DataLoader object will be unhappy will the
        # default parameters

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
                patch('trojai.modelgen.default_optimizer.train_val_dataset_split',
                      return_value=(dataset, dataset)) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                train_acc_noop = 1.0
                train_loss_noop = 1.0
                ts = EpochTrainStatistics(train_acc_noop, train_loss_noop)
                val_acc_noop = 1.0
                if epoch < 2:
                    val_loss = 10.0 - epoch  # we keep the loss decreasing until the first 4 epochs
                    # This prevents the early-stopping code from being activated,
                    # since the loss is decreasing every epoch
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs
                else:
                    val_loss = 9.0 - eps  # decrease the loss, but only by eps, so we quit
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs

            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # TODO: explain why this shoudl be 8
            self.assertEqual(8, num_epochs_trained)

    def test_early_stopping2(self):
        """
        The purpose of this test is to ensure that the early stopping is activated.  The test works by Mocking the
        train_epoch function.  After 2 epochs, the train_epoch function returns BatchStatistics with val_loss
        increasing and thus we should stop at 8 epochs of training
        """
        optimizer = DefaultOptimizer()
        optimizer.optimizer_cfg.training_cfg.epochs = 10

        optimizer.optimizer_cfg.training_cfg.early_stopping = EarlyStoppingConfig()
        model = Mock(spec=nn.Module)
        model.parameters = Mock()
        dataset = Mock(spec=torch.utils.data.BatchSampler)
        dataset.__len__ = Mock(return_value=2)  # otherwise the pytorch DataLoader object will be unhappy will the
        # default parameters

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
                patch('trojai.modelgen.default_optimizer.train_val_dataset_split',
                      return_value=(dataset, dataset)) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                train_acc_noop = 1.0
                train_loss_noop = 1.0
                ts = EpochTrainStatistics(train_acc_noop, train_loss_noop)
                val_acc_noop = 1.0
                if epoch < 2:
                    val_loss = 10.0 - epoch  # we keep the loss decreasing until the first 4 epochs
                    # This prevents the early-stopping code from being activated,
                    # since the loss is decreasing every epoch
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs
                else:
                    val_loss = float(epoch)  # we fix the loss from here on within eps,
                    # we expect it to quit in 5 epochs
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs

            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # TODO: explain why answer is 8
            self.assertEqual(9, num_epochs_trained)

    def test_early_stopping3(self):
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
        dataset = Mock(spec=torch.utils.data.BatchSampler)
        dataset.__len__ = Mock(return_value=2)  # otherwise the pytorch DataLoader object will be unhappy will the
        # default parameters

        # patch disables import torch.optim, so we can skip creating models to test the optimizer
        with patch('trojai.modelgen.default_optimizer.torch.optim.Adam') as patched_optimizer, \
                patch('trojai.modelgen.default_optimizer.train_val_dataset_split',
                      return_value=(dataset, dataset)) as patched_train_val_split:

            # this function overrides the return value of train_epoch, so that we can simulate
            # when early-stopping is supposed to occur, and
            def train_epoch_side_effect(net, train_loader, val_loader, epoch, progress_bar_disable=True):
                # these variables are not consequential for the early-stopping code, so we just set them to
                # constants
                train_acc_noop = 1.0
                train_loss_noop = 1.0
                ts = EpochTrainStatistics(train_acc_noop, train_loss_noop)
                val_acc_noop = 1.0
                if epoch < 2:
                    val_loss = 10.0 - epoch  # we keep the loss decreasing until the first 4 epochs
                    # This prevents the early-stopping code from being activated,
                    # since the loss is decreasing every epoch
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs
                else:
                    val_loss = float(epoch)  # we fix the loss from here on within eps,
                    # we expect it to quit in 5 epochs
                    vs = EpochValidationStatistics(val_acc_noop, val_loss)
                    return ts, vs

            optimizer.train_epoch = Mock(side_effect=train_epoch_side_effect)
            _, _, num_epochs_trained = optimizer.train(model, dataset)
            # the early stopping should *not* have been run, b/c we set it to None, so we should
            # have trained for the full 10 epochs
            self.assertEqual(num_epochs_trained, optimizer.optimizer_cfg.training_cfg.epochs)

    # TODO: add mock tests on saving best model


if __name__ == '__main__':
    unittest.main()
