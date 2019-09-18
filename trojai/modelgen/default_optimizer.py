import collections
import logging
import os
from typing import Sequence
import copy
import cloudpickle as pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .training_statistics import BatchStatistics, EpochStatistics
from .optimizer_interface import OptimizerInterface
from .config import DefaultOptimizerConfig
from .constants import VALID_OPTIMIZERS

logger = logging.getLogger(__name__)


def _eval_acc(y_hat: torch.Tensor, y_truth: torch.Tensor, n_total: int = 0, n_correct: int = 0):
    """
    Wrapper for computing accuracy
    :param y_hat: the computed predictions, should be of shape (n_batches, num_classes)
    :param y_truth: the actual y-values
    :param n_total: the total number of data points processed, this will be incremented and returned
    :param n_correct: the total number of correct predictions so far, before this function was called
    :return: accuracy, updated n_total, updated n_correct

    TODO:
     [ ] - need to handle the case where the user applies sigmoid at the output of the final layer before
           outputting.  With the current behavior, _eval_acc would apply the sigmoid function twice
     [ ] - are there non-sigmoid conversions we'd want to support when rounding predictions w/ one output class?
    """
    y_hat_size = y_hat.size()
    if len(y_hat_size) == 2:
        n_batches, num_classes = y_hat.size()
    elif len(y_hat_size) == 1:
        n_batches = y_hat_size[0]
        num_classes = 1
    else:
        msg = "unhandled size of y_hat!:" + str(y_hat.size())
        logger.error(msg)
        raise ValueError(msg)
    n_total += n_batches
    if num_classes > 1:
        max_index = y_hat.max(dim=1)[1]
        n_correct += (max_index == y_truth).sum().item()
    else:
        rounded_preds = torch.round(torch.sigmoid(y_hat))
        n_correct += (rounded_preds.int() == y_truth.int()).sum().item()
    acc = 100. * float(n_correct) / float(n_total)
    return acc, n_total, n_correct


def train_val_dataset_split(dataset: torch.utils.data.Dataset, split_amt: float) \
        -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Splits a PyTorch dataset (of type: torch.utils.data.Dataset) into train/test
    TODO:
      [ ] - specify random seed to torch splitter
    :param dataset: the dataset to be split
    :param split_amt: fraction specificing the validation dataset size relative to the whole.  1-split_amt will
                      be the size of the training dataset
    :return: a tuple of the train and validation datasets
    """

    if split_amt < 0 or split_amt > 1:
        msg = "Dataset split amount must be between 0 and 1"
        logger.error(msg)
        raise ValueError(msg)

    dataset_len = len(dataset)
    train_len = int(dataset_len * (1 - split_amt))
    val_len = int(dataset_len - train_len)
    lengths = [train_len, val_len]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
    return train_dataset, val_dataset


class DefaultOptimizer(OptimizerInterface):
    """
    Defines the default optimizer which trains the models
    """

    def __init__(self, optimizer_cfg: DefaultOptimizerConfig = None):
        """
        Initializes the default optimizer with a DefaultOptimizerConfig
        :param optimizer_cfg: the configuration used to initialize the DefaultOptimizer
        """
        if optimizer_cfg is None:
            logger.info("Using default parameters to setup Optimizer!")
            self.optimizer_cfg = DefaultOptimizerConfig()
        elif not isinstance(optimizer_cfg, DefaultOptimizerConfig):
            msg = "optimizer_cfg must be of type DefaultOptimizerConfig"
            logger.error(msg)
            raise TypeError(msg)
        else:
            self.optimizer_cfg = optimizer_cfg

        # setup parameters for training here
        self.device = self.optimizer_cfg.training_cfg.device

        self.loss_function_str = self.optimizer_cfg.training_cfg.objective.lower()
        if self.loss_function_str == "cross_entropy_loss".lower():
            self.loss_function = nn.CrossEntropyLoss()
        elif self.loss_function_str == 'BCEWithLogitsLoss'.lower():
            self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_function.to(self.device)

        self.lr = self.optimizer_cfg.training_cfg.lr
        self.optimizer_str = self.optimizer_cfg.training_cfg.optim.lower()
        self.optimizer = None

        self.batch_size = self.optimizer_cfg.training_cfg.batch_size
        self.num_epochs = self.optimizer_cfg.training_cfg.epochs
        self.save_best_model = self.optimizer_cfg.training_cfg.save_best_model

        self.str_description = "{'batch_size':%d, 'num_epochs':%d, 'device':'%s', 'lr':%.5e, 'loss_function':'%s', " \
                               "'optimizer':'%s'}" % \
                               (self.batch_size, self.num_epochs, self.device.type, self.lr, self.loss_function_str,
                                self.optimizer_str)

        # setup configuration for logging and tensorboard
        self.num_batches_per_logmsg = self.optimizer_cfg.reporting_cfg.num_batches_per_logmsg
        self.num_epochs_per_metrics = self.optimizer_cfg.reporting_cfg.num_epochs_per_metrics
        self.num_batches_per_metrics = self.optimizer_cfg.reporting_cfg.num_batches_per_metrics
        # NOTE: the configuration parameter 'num_batches_per_val_dataset_metrics' has the most significant impact on
        #  training performance.  It can be useful for debugging development of a model, but when scaling up,
        #  set this parameter to None to disable computing the validation dataset and speed up training
        self.num_batches_per_val_dataset_metrics = self.optimizer_cfg.reporting_cfg.num_batches_per_val_dataset_metrics
        if self.device.type == 'cpu' and self.num_batches_per_val_dataset_metrics is not None:
            logger.warning('Training will be VERY SLOW on a CPU with num_batches_per_val_dataset_metrics set to a '
                           'value other than None.  If validation dataset metrics are still desired, '
                           'consider increasing this value to speed up training')

        tensorboard_output_dir = self.optimizer_cfg.reporting_cfg.tensorboard_output_dir
        self.tb_writer = None
        if tensorboard_output_dir is not None:
            self.tb_writer = SummaryWriter(tensorboard_output_dir)

        optimizer_cfg_str = 'Optimizer[%s] Configured as: loss[%s], learning-rate[%.5e], batch-size[%d] ' \
                            'num-epochs[%d] Device[%s]' % \
                            (self.optimizer_str, str(self.loss_function), self.lr, self.batch_size, self.num_epochs,
                             self.device.type)
        reporting_cfg_str = 'Reporting Configured as: num_batches_per_log_message[%d] tensorboard_dir[%s]' % \
                            (self.num_batches_per_logmsg, tensorboard_output_dir)
        metrics_capture_str = 'Metrics capturing configured as: num_epochs_per_metric[%d] ' \
                              'num_batches_per_epoch_per_metric[%d]' % \
                              (self.num_epochs_per_metrics, self.num_batches_per_metrics)
        logger.info(self.str_description)
        logger.info(optimizer_cfg_str)
        logger.info(reporting_cfg_str)
        logger.info(metrics_capture_str)

    def __str__(self) -> str:
        return self.str_description

    def __deepcopy__(self, memodict={}):
        optimizer_cfg_copy = copy.deepcopy(self.optimizer_cfg)
        # WARNING: this assumes that none of the derived attributes have been changed after construction!
        return DefaultOptimizer(DefaultOptimizerConfig(optimizer_cfg_copy.training_cfg,
                                                       optimizer_cfg_copy.reporting_cfg))

    def get_cfg_as_dict(self) -> dict:
        output_dict = dict(device=str(self.device.type),
                           epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           learning_rate=self.lr,
                           optim=self.optimizer_str,
                           objective=self.loss_function_str)
        return output_dict

    def __eq__(self, other) -> bool:
        try:
            if self.optimizer_cfg == other.optimizer_cfg:
                # we still check the derived attributes to ensure that they remained the same after
                # after construction
                if self.device.type == other.device.type and self.loss_function_str == other.loss_function_str and \
                        self.lr == other.lr and self.optimizer_str == other.optimizer_str and \
                        self.batch_size == other.batch_size and self.num_epochs == other.num_epochs and \
                        self.str_description == other.str_description and \
                        self.num_batches_per_logmsg == other.num_batches_per_logmsg and \
                        self.num_epochs_per_metrics == other.num_epochs_per_metrics and \
                        self.num_batches_per_metrics == other.num_batches_per_metrics and \
                        self.num_batches_per_val_dataset_metrics == other.num_batches_per_val_dataset_metrics:
                    if self.tb_writer is not None:
                        if other.tb_writer is not None:
                            if self.tb_writer.log_dir == other.tb_writer.log_dir:
                                return True
                            else:
                                return False
                        else:
                            return False
                    else:
                        if other.tb_writer is not None:
                            return False
                        else:
                            # both are None
                            return True
            else:
                return False
        except AttributeError:
            return False

    def get_device_type(self) -> str:
        """
        :return: a string representing the device used to train the model
        """
        return self.device.type

    def save(self, fname: str) -> None:
        """
        Saves the configuration object used to construct the DefaultOptimizer.
        NOTE: because the DefaultOptimizer object itself is not persisted, but rather the
          DefaultOptimizerConfig object, the state of the object is not persisted!
        :param fname: the filename to save the DefaultOptimizer's configuration.
        :return: None
        """
        self.optimizer_cfg.save(fname)

    @staticmethod
    def load(fname: str) -> OptimizerInterface:
        """
        Reconstructs a DefaultOptimizer, by loading the configuration used to construct the original
        DefaultOptimizer, and then creating a new DefaultOptimizer object from the saved configuration
        :param fname: The filename of the saved optimzier
        :return: a DefaultOptimizer object
        """
        with open(fname, 'rb') as f:
            loaded_optimzier_cfg = pickle.load(f)
        return DefaultOptimizer(loaded_optimzier_cfg)

    def _eval_loss_function(self, y_hat: torch.Tensor, y_truth: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for evaluating the loss function to abstract out any data casting we need to do
        :param y_hat: the predicted y-value
        :param y_truth: the actual y-value
        :return: the loss associated w/ the prediction and actual
        """
        if self.loss_function_str == "cross_entropy_loss":
            train_loss = self.loss_function(y_hat, y_truth.long())
        else:
            train_loss = self.loss_function(y_hat, y_truth)
        return train_loss

    def train(self, net: torch.nn.Module, dataset: torch.utils.data.Dataset, train_val_split: float = 0.0,
              progress_bar_disable: bool = False, torch_dataloader_kwargs: dict = None) \
        -> (torch.nn.Module, Sequence[EpochStatistics]):
        """
        Train the network.
        :param net: the network to train
        :param dataset: the dataset to train the network on
        :param train_val_split: the % of training data to use as validation data
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: any additional kwargs to pass to PyTorch's native DataLoader
        :return: the trained network, and a list of EpochStatistics objects which contain the statistics for training
        """
        net = net.to(self.device)

        net.train()  # put network into training mode
        if self.optimizer_str == 'adam':
            self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        elif self.optimizer_str not in VALID_OPTIMIZERS:
            msg = self.optimizer_str + " is not a supported optimizer!"
            logger.error(msg)
            raise ValueError(msg)
        else:
            msg = self.optimizer_str + " not yet implemented!"
            logger.error(msg)
            raise NotImplementedError(msg)

        # set according to the following guidelines:
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        # https://pytorch.org/docs/stable/notes/cuda.html
        pin_memory = False
        if self.device.type != 'cpu':
            pin_memory = True

        # split into train & validation datasets, and setup data loaders
        data_loader_kwargs_in = {} if torch_dataloader_kwargs is None else torch_dataloader_kwargs
        logger.info('DataLoader[Train/Val] kwargs=' + str(torch_dataloader_kwargs))

        train_dataset, val_dataset = train_val_dataset_split(dataset, train_val_split)
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True,
                                  **data_loader_kwargs_in)
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True,
                                **data_loader_kwargs_in)

        # use validation in training? provide as option?
        all_epochs_stats = []
        best_net = None
        best_validation_acc = -999
        best_training_acc = -999
        for epoch_idx, epoch in enumerate(range(self.num_epochs)):
            compute_batch_stats = True if epoch_idx % self.num_epochs_per_metrics == 0 else False
            batches_stats = self.train_epoch(net, train_loader, val_loader, epoch, compute_batch_stats,
                                             progress_bar_disable=progress_bar_disable)

            if compute_batch_stats and len(batches_stats) > 0:
                epoch_training_stats = EpochStatistics(epoch_idx)
                epoch_training_stats.add_batch(batches_stats)
                all_epochs_stats.append(epoch_training_stats)

                if self.save_best_model:
                    if train_val_split == 0.0:
                        # use training accuracy as the metric for deciding the best model
                        final_batch_training_acc = batches_stats[-1].batch_train_accuracy
                        if final_batch_training_acc >= best_training_acc:
                            msg = "Updating best model with epoch:[%d] accuracy[%0.02f].  Previous best training " \
                                  "accuracy was: %0.02f" % (epoch_idx, final_batch_training_acc, best_training_acc)
                            logger.info(msg)
                            best_net = net
                            best_training_acc = final_batch_training_acc
                    else:
                        # use validation accuracy as the metric for deciding the best model
                        final_batch_validation_acc = batches_stats[-1].batch_validation_accuracy
                        if final_batch_validation_acc >= best_validation_acc:
                            msg = "Updating best model with epoch:[%d] accuracy[%0.02f].  Previous best validation " \
                                  "accuracy was: %0.02f" % (epoch_idx, final_batch_validation_acc, best_validation_acc)
                            logger.info(msg)
                            best_net = net
                            best_validation_acc = final_batch_validation_acc

        if self.save_best_model:
            return best_net, all_epochs_stats
        else:
            return net, all_epochs_stats

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    epoch_num: int, compute_batch_stats: bool = True,
                    avg_loss_num_batches: int = 100, progress_bar_disable: bool = False):
        """
        Runs one epoch of training on the specified model
        TODO:
          [ ] - pass in the avg_loss_num_batches parameter cleanly and configurably

        :param model: the model to train for one epoch
        :param train_loader: a DataLoader object pointing to the training dataset
        :param val_loader: a DataLoader object pointing to the validation dataset
        :param epoch_num: the epoch number that is being trained
        :param compute_batch_stats: if True, computes statistics for the batch based on the reporting configuration
                specified in the initialization of the optimizer
        :param avg_loss_num_batches: the number of batches of data to accumulate to compute average loss
        :param progress_bar_disable: if True, disables the progress bar
        :return: a list of statistics for batches where statistics were computed
        """
        loop = tqdm(train_loader, disable=progress_bar_disable)

        pid = os.getpid()
        train_dataset_len = len(train_loader.dataset)
        train_loader_len = len(train_loader)

        # NOTE: potential speed-up by not computing the average ... but this seems like premature optimization to me
        avg_train_loss_circbuf = collections.deque(maxlen=avg_loss_num_batches)
        avg_val_loss_vec = np.empty(len(val_loader.dataset))

        train_n_correct, train_n_total = 0, 0
        val_n_correct, val_n_total = 0, 0
        val_acc, val_loss = None, None
        batch_stats = []
        for batch_idx, (x, y_truth) in enumerate(loop):
            x = x.to(self.device)
            y_truth = y_truth.to(self.device)

            # put network into training mode & zero out previous gradient computations
            model.train()
            self.optimizer.zero_grad()

            # get predictions based on input & weights learned so far
            y_hat = model(x)

            # compute metrics
            batch_train_loss = self._eval_loss_function(y_hat, y_truth)
            running_train_acc, train_n_total, train_n_correct = _eval_acc(y_hat, y_truth,
                                                                          n_total=train_n_total,
                                                                          n_correct=train_n_correct)

            # compute gradient
            batch_train_loss.backward()
            self.optimizer.step()

            if len(val_loader) > 0 and self.num_batches_per_val_dataset_metrics is not None and \
                    ((batch_idx % self.num_batches_per_val_dataset_metrics == 0) or
                     (batch_idx % self.num_batches_per_metrics == 0)):
                # last condition ensures metrics are computed for storage put model into evaluation mode
                model.eval()
                # turn off auto-grad for validation set computation
                with torch.no_grad():
                    for val_batch_idx, (x_eval, y_truth_eval) in enumerate(val_loader):
                        x_eval = x_eval.to(self.device)
                        y_truth_eval = y_truth_eval.to(self.device)
                        y_hat_eval = model(x_eval)

                        val_loss_tensor = self._eval_loss_function(y_hat_eval, y_truth_eval)
                        val_loss = val_loss_tensor.item()
                        val_acc, val_n_total, val_n_correct = _eval_acc(y_hat_eval, y_truth_eval,
                                                                        n_total=val_n_total,
                                                                        n_correct=val_n_correct)
                        avg_val_loss_vec[val_batch_idx] = val_loss

                avg_val_loss = np.mean(avg_val_loss_vec)
                if self.tb_writer is not None:
                    try:
                        self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                                  '-avg_validation_loss', avg_val_loss)
                    except:
                        # TODO: catch specific expcetions
                        pass

            # NOTE: should these be options to compute training accuracy, etc, as well?
            avg_train_loss_circbuf.append(batch_train_loss.item())
            avg_train_loss = np.mean(avg_train_loss_circbuf)

            loop.set_description('Epoch {}/{}'.format(epoch_num + 1, self.num_epochs))
            loop.set_postfix(avg_train_loss=avg_train_loss)

            # report batch statistics to tensorflow
            if self.tb_writer is not None:
                try:
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-train_loss',
                                              batch_train_loss.item())
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-avg_train_loss',
                                              avg_train_loss)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-running_train_acc',
                                              running_train_acc)
                    if len(val_loader) > 0 and self.num_batches_per_val_dataset_metrics is not None:
                        self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-val_acc', val_acc)
                except:
                    # TODO: catch specific expcetions
                    pass

            # save batch statistics
            if compute_batch_stats and (batch_idx % self.num_batches_per_metrics == 0):
                batch_stat = BatchStatistics(batch_idx, running_train_acc, batch_train_loss.item(),
                                             val_acc, val_loss)
                batch_stats.append(batch_stat)

            if batch_idx % self.num_batches_per_logmsg == 0:
                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tAvgTrainLoss: {:.6f}\tTrainAcc: {:.6f}'.format(
                    pid, epoch_num, batch_idx * len(x), train_dataset_len,
                                    100. * batch_idx / train_loader_len, avg_train_loss, running_train_acc))

        return batch_stats

    def test(self, net: nn.Module, clean_data: Dataset, triggered_data: Dataset,
             progress_bar_disable: bool = False, torch_dataloader_kwargs: dict = None) -> dict:
        """
        Test the trained network
        :param net: the trained module to run the test data through
        :param clean_data: the clean Dataset
        :param triggered_data: the triggered Dataset, if None, not computed
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: any keyword arguments to pass directly to PyTorch's DataLoader
        :return: a dictionary of the statistics on the clean and triggered data (if applicable)
        """
        test_data_statistics = {}
        net.eval()
        data_loader_kwargs_in = {} if torch_dataloader_kwargs is None else torch_dataloader_kwargs
        logger.info('DataLoader[Test] kwargs=' + str(torch_dataloader_kwargs))

        pin_memory = False
        if self.device.type != 'cpu':
            pin_memory = True
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        data_loader = DataLoader(clean_data, batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True,
                                 **data_loader_kwargs_in)

        # test type is classification accuracy on clean and triggered data
        test_n_correct = 0
        test_n_total = 0
        with torch.no_grad():
            for batch, (x, y_truth) in enumerate(data_loader):
                x = x.to(self.device)
                y_truth = y_truth.to(self.device)
                y_hat = net(x)
                test_acc, test_n_total, test_n_correct = _eval_acc(y_hat, y_truth,
                                                                   n_total=test_n_total,
                                                                   n_correct=test_n_correct)
        test_data_statistics['clean_accuracy'] = test_acc
        test_data_statistics['clean_n_total'] = test_n_total
        logger.info("Accuracy on clean test data: %0.02f" %
                    (test_data_statistics['clean_accuracy'],))

        if triggered_data is None:
            return test_data_statistics

        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        data_loader = DataLoader(triggered_data, batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True)
        test_n_correct = 0
        test_n_total = 0
        with torch.no_grad():
            for batch, (x, y_truth) in enumerate(data_loader):
                x = x.to(self.device)
                y_truth = y_truth.to(self.device)
                y_hat = net(x)
                test_acc, test_n_total, test_n_correct = _eval_acc(y_hat, y_truth,
                                                                   n_total=test_n_total,
                                                                   n_correct=test_n_correct)
        test_data_statistics['triggered_accuracy'] = test_acc
        test_data_statistics['triggered_n_total'] = test_n_total
        logger.info("Accuracy on triggered test data: %0.02f" %
                    (test_data_statistics['triggered_accuracy'],))
        return test_data_statistics
