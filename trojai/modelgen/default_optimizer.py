import logging
import os
from typing import Sequence, Callable
import copy
import cloudpickle as pickle
from collections import defaultdict

import datetime
import tempfile
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.clip_grad as torch_clip_grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .training_statistics import EpochStatistics, EpochValidationStatistics, EpochTrainStatistics
from .optimizer_interface import OptimizerInterface
from .config import DefaultOptimizerConfig, DefaultSoftToHardFn, default_soft_to_hard_fn_kwargs
from .constants import VALID_OPTIMIZERS, MAX_EPOCHS
from .datasets import CSVDataset

logger = logging.getLogger(__name__)


def _eval_acc(y_hat: torch.Tensor, y_truth: torch.Tensor,
    n_total: defaultdict = None, n_correct: defaultdict = None,
    soft_to_hard_fn: Callable = None,
    soft_to_hard_fn_kwargs: dict = None):
    # TODO: make soft_to_hard_fn and soft_to_hard_fn_kwargs configurable through config interface

    """
    Wrapper for computing accuracy in an on-line manner
    :param y_hat: the computed predictions, should be of shape (n_batches, num_output_neurons)
    :param y_truth: the actual y-values
    :param n_total: a defaultdict with keys representing labels, and values representing the # of times examples
        with that label have been seen.  Example: {0: 10, 1: 20, 2: 5, 3: 30}
    :param n_correct: a defaultdict with keys representing labels, and values representing the # of times examples
        with that label have been corrected.  Example: {0: 8, 1: 15, 2: 5, 3: 25}
    :return: accuracy, updated n_total, updated n_correct
    """

    y_hat_size = y_hat.size()
    if len(y_hat_size) == 2:
        num_output_neurons = y_hat_size[1]
    elif len(y_hat_size) == 1:
        num_output_neurons = 1
    else:
        msg = "unsupported size of y_hat!:" + str(y_hat.size())
        logger.error(msg)
        raise ValueError(msg)

    if not soft_to_hard_fn:
        soft_to_hard_fn = DefaultSoftToHardFn()
    if not soft_to_hard_fn_kwargs:
        soft_to_hard_fn_kwargs = copy.deepcopy(default_soft_to_hard_fn_kwargs)

    # increment n_total per class
    label, unique_counts = y_truth.unique(return_counts=True)
    if not n_total:
        n_total = defaultdict(int)
    for ii, k in enumerate(label):
        n_total[k.item()] += unique_counts[ii].item()

    if not n_correct:
        n_correct = defaultdict(int)

    hard_decision_pred = soft_to_hard_fn(y_hat, **soft_to_hard_fn_kwargs)
    label, n_correct_per_class = hard_decision_pred[hard_decision_pred == y_truth.int()].unique(return_counts=True)
    for ii, k in enumerate(label):
        n_correct[k.item()] += n_correct_per_class[ii].item()

    acc = 0.
    weight = 1. / len(n_total.keys())
    for k in n_total.keys():
        acc += 0 if n_total[k] == 0 else float(n_correct[k]) / float(n_total[k])

    acc *= 100. * weight

    return acc, n_total, n_correct


def _save_nandata(x, y_hat, y_truth, loss_tensor, loss_val, acc_val, n_total, n_correct, model):
    """

    """
    save_folder = tempfile.mkdtemp()

    dict_to_save = dict(y_hat=y_hat,
                        y_truth=y_truth,
                        x=x,
                        loss_tensor=loss_tensor,
                        n_total=n_total,
                        n_correct=n_correct,
                        loss_val=loss_val,
                        acc_val=acc_val)

    try:
        os.makedirs(save_folder)
    except IOError:
        pass
    t = str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace('-', '_').replace(' ', '_')
    with open(os.path.join(save_folder, 'data_' + t + '.pkl'), 'wb') as f:
        pickle.dump(dict_to_save, f)
    torch.save(model, os.path.join(save_folder, 'model_' + t + '.pkl'))

    msg = "Loss function and/or _eval_acc returned NaN while training! " \
          "This usually means gradient explosion.  " \
          "Try turning on gradient clipping and/or learning rate scheduling.  Check the log" \
          "files for more information, and the folder: " + str(save_folder)
    logger.error(msg)
    sys.exit()


def train_val_dataset_split(dataset: torch.utils.data.Dataset, split_amt: float, val_data_transform: Callable,
    val_label_transform: Callable) \
    -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Splits a PyTorch dataset (of type: torch.utils.data.Dataset) into train/test
    TODO:
      [ ] - specify random seed to torch splitter
    :param dataset: the dataset to be split
    :param split_amt: fraction specificing the validation dataset size relative to the whole.  1-split_amt will
                      be the size of the training dataset
    :param val_data_transform: (function: any -> any) how to transform the validation data to fit
            into the desired model and objective function
    :param val_label_transform: (function: any -> any) how to transform the validation labels
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
    val_dataset.data_transform = val_data_transform
    val_dataset.label_transform = val_label_transform
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

        if not callable(self.optimizer_cfg.training_cfg.objective):
            self.loss_function_str = self.optimizer_cfg.training_cfg.objective.lower()
            if self.loss_function_str == "cross_entropy_loss".lower():
                self.loss_function = nn.CrossEntropyLoss(**self.optimizer_cfg.training_cfg.objective_kwargs)
            elif self.loss_function_str == 'BCEWithLogitsLoss'.lower():
                self.loss_function = nn.BCEWithLogitsLoss(**self.optimizer_cfg.training_cfg.objective_kwargs)
            else:
                msg = self.loss_function_str + ": Unsupported objective function!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            self.loss_function = self.optimizer_cfg.training_cfg.objective
        self.loss_function.to(self.device)
        self.soft_to_hard_fn = self.optimizer_cfg.training_cfg.soft_to_hard_fn
        self.soft_to_hard_fn_kwargs = self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs

        self.lr = self.optimizer_cfg.training_cfg.lr
        # setup learning rate scheduler if desired
        if self.optimizer_cfg.training_cfg.lr_scheduler is not None:
            self.lr_scheduler = self.optimizer_cfg.training_cfg.lr_scheduler(self.optimizer,
                                                                             **self.optimizer_cfg.training_cfg.lr_scheduler_init_kwargs)

        self.optimizer_str = self.optimizer_cfg.training_cfg.optim.lower()
        self.optimizer = None
        self.optim_kwargs = self.optimizer_cfg.training_cfg.optim_kwargs

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

        # raise error if train/val split is not set properly for either saving best model
        # or for early stopping
        if self.optimizer_cfg.training_cfg.early_stopping or self.save_best_model:
            self.num_epochs_per_metrics = 1
            logger.warning("Overriding num_epochs_per_metrics due to early-stopping or saving-best-model!")

        if self.device.type == 'cpu' and self.num_batches_per_metrics:
            logger.warning('Training will be VERY SLOW on a CPU with num_batches_per_metrics set to a '
                           'value other than None.  If validation dataset metrics are still desired, '
                           'consider increasing this value to speed up training')

        tensorboard_output_dir = self.optimizer_cfg.reporting_cfg.tensorboard_output_dir
        self.tb_writer = None
        if tensorboard_output_dir:
            self.tb_writer = SummaryWriter(tensorboard_output_dir)

        optimizer_cfg_str = 'Optimizer[%s] Configured as: loss[%s], learning-rate[%.5e], batch-size[%d] ' \
                            'num-epochs[%d] Device[%s]' % \
                            (self.optimizer_str, str(self.loss_function), self.lr, self.batch_size, self.num_epochs,
                             self.device.type)
        reporting_cfg_str = 'Reporting Configured as: num_batches_per_log_message[%d] tensorboard_dir[%s]' % \
                            (self.num_batches_per_logmsg, tensorboard_output_dir)
        nbpm_print = self.num_batches_per_metrics if self.num_batches_per_metrics else -1
        metrics_capture_str = 'Metrics capturing configured as: num_epochs_per_metric[%d] ' \
                              'num_batches_per_epoch_per_metric[%d]' % \
                              (self.num_epochs_per_metrics, nbpm_print)
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
        return self.optimizer_cfg.training_cfg.get_cfg_as_dict()

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
                    self.num_batches_per_metrics == other.num_batches_per_metrics:
                    if self.tb_writer:
                        if other.tb_writer:
                            if self.tb_writer.log_dir == other.tb_writer.log_dir:
                                return True
                            else:
                                return False
                        else:
                            return False
                    else:
                        if other.tb_writer:
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

    def train(self, net: torch.nn.Module, dataset: CSVDataset, progress_bar_disable: bool = False,
        torch_dataloader_kwargs: dict = None) -> (torch.nn.Module, Sequence[EpochStatistics], int):
        """
        Train the network.
        :param net: the network to train
        :param dataset: the dataset to train the network on
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: any additional kwargs to pass to PyTorch's native DataLoader
        :return: the trained network, and a list of EpochStatistics objects which contain the statistics for training,
                and the # of epochs on which the net was trained
        """
        net = net.to(self.device)

        net.train()  # put network into training mode
        if self.optimizer_str == 'adam':
            self.optimizer = optim.Adam(net.parameters(), lr=self.lr, **self.optim_kwargs)
        elif self.optimizer_str == 'sgd':
            self.optimizer = optim.SGD(net.parameters(), lr=self.lr, **self.optim_kwargs)
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
        data_loader_kwargs_in = dict(batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True,
                                     shuffle=True)
        if torch_dataloader_kwargs:
            data_loader_kwargs_in.update(torch_dataloader_kwargs)

        val_data_loader_kwargs_in = dict(batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True,
                                         shuffle=True)
        if self.optimizer_cfg.training_cfg.val_dataloader_kwargs:
            val_data_loader_kwargs_in.update(self.optimizer_cfg.training_cfg.val_dataloader_kwargs)
        if torch_dataloader_kwargs:
            val_data_loader_kwargs_in.update(torch_dataloader_kwargs)

        logger.info('DataLoader[Train/Val] kwargs=' + str(torch_dataloader_kwargs))

        train_dataset, val_dataset = train_val_dataset_split(dataset, self.optimizer_cfg.training_cfg.train_val_split,
                                                             self.optimizer_cfg.training_cfg.val_data_transform,
                                                             self.optimizer_cfg.training_cfg.val_label_transform)
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        train_loader = DataLoader(train_dataset, **data_loader_kwargs_in)
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        val_loader = DataLoader(val_dataset, **val_data_loader_kwargs_in) if len(val_dataset) > 0 else []

        # stores training & val data statistics for every epoch
        epoch_stats = []
        best_net = None
        best_validation_acc = -999
        best_val_loss = np.inf
        best_val_loss_epoch = -1

        num_epochs_to_monitor = 1
        if self.optimizer_cfg.training_cfg.early_stopping:
            num_epochs_to_monitor = self.optimizer_cfg.training_cfg.early_stopping.num_epochs

        epoch = 0
        done = False
        while not done:
            train_stats, validation_stats = self.train_epoch(net, train_loader, val_loader, epoch,
                                                             progress_bar_disable=progress_bar_disable)
            epoch_training_stats = EpochStatistics(epoch, train_stats, validation_stats)
            epoch_stats.append(epoch_training_stats)

            # TODO: save best model should use same criterion as early stopping (val-loss rather than val-acc)?
            if self.save_best_model:
                # use validation accuracy as the metric for deciding the best model
                if validation_stats.val_acc >= best_validation_acc:
                    msg = "Updating best model with epoch:[%d] accuracy[%0.02f].  Previous best validation " \
                          "accuracy was: %0.02f" % (epoch, validation_stats.val_acc, best_validation_acc)
                    logger.info(msg)
                    best_net = copy.deepcopy(net)
                    best_validation_acc = validation_stats.val_acc

            # early stopping
            # record the val loss of the last batch in the epoch.  if N epochs after the best val_loss, we have not
            # improved the val-loss by atleast eps, we quit
            if self.optimizer_cfg.training_cfg.early_stopping:
                # EarlyStoppingConfig validates that eps > 0 as well ..
                if validation_stats.val_loss < (
                    best_val_loss - np.abs(self.optimizer_cfg.training_cfg.early_stopping.val_loss_eps)):
                    best_val_loss = validation_stats.val_loss
                    best_val_loss_epoch = epoch
                    best_net = copy.deepcopy(net)
                    logger.info('EarlyStopping - NewBest >> best_val_loss:%0.04f best_val_loss_epoch:%d' %
                                (best_val_loss, best_val_loss_epoch))
                elif epoch >= (best_val_loss_epoch + num_epochs_to_monitor):
                    epoch += 1  # we do this b/c of the break to keep the accounting of epoch # returned to
                    # the user to be one based
                    msg = "Exiting training loop in epoch: %d - due to early stopping criterion being met!" \
                          % (epoch,)
                    logger.warning(msg)
                    done = True

            epoch += 1
            if self.optimizer_cfg.training_cfg.early_stopping:
                # in case something goes wrong, we exit after training a long time ...
                if epoch >= MAX_EPOCHS:
                    done = True
            else:
                if epoch >= self.num_epochs:
                    done = True

        if self.save_best_model or self.optimizer_cfg.training_cfg.early_stopping:
            return best_net, epoch_stats, epoch
        else:
            return net, epoch_stats, epoch

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        epoch_num: int, progress_bar_disable: bool = False):
        """
        Runs one epoch of training on the specified model

        :param model: the model to train for one epoch
        :param train_loader: a DataLoader object pointing to the training dataset
        :param val_loader: a DataLoader object pointing to the validation dataset
        :param epoch_num: the epoch number that is being trained
        :param progress_bar_disable: if True, disables the progress bar
        :return: a list of statistics for batches where statistics were computed
        """

        pid = os.getpid()
        train_dataset_len = len(train_loader.dataset)
        loop = tqdm(train_loader, disable=progress_bar_disable)

        train_n_correct, train_n_total = None, None
        val_n_correct, val_n_total = None, None
        sum_batchmean_train_loss = 0
        running_train_acc = 0
        num_batches = len(train_loader)
        for batch_idx, (x, y_truth) in enumerate(loop):
            x = x.to(self.device)
            y_truth = y_truth.to(self.device)

            # put network into training mode & zero out previous gradient computations
            model.train()  # TODO: this should be outside of the loop
            self.optimizer.zero_grad()

            # get predictions based on input & weights learned so far
            y_hat = model(x)

            # compute metrics
            batch_train_loss = self._eval_loss_function(y_hat, y_truth)
            sum_batchmean_train_loss += batch_train_loss.item()

            running_train_acc, train_n_total, train_n_correct = _eval_acc(y_hat, y_truth,
                                                                          n_total=train_n_total,
                                                                          n_correct=train_n_correct,
                                                                          soft_to_hard_fn=self.soft_to_hard_fn,
                                                                          soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)

            if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
                _save_nandata(x, y_hat, y_truth, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
                              train_n_total, train_n_correct, model)

            # compute gradient
            batch_train_loss.backward()

            # perform gradient clipping if configured
            if self.optimizer_cfg.training_cfg.clip_grad:
                if self.optimizer_cfg.training_cfg.clip_type == 'norm':
                    # clip_grad_norm_ modifies gradients in place
                    #  see: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
                    torch_clip_grad.clip_grad_norm_(model.parameters(), self.optimizer_cfg.training_cfg.clip_val,
                                                    **self.optimizer_cfg.training_cfg.clip_kwargs)
                elif self.optimizer_cfg.training_cfg.clip_type == 'val':
                    # clip_grad_val_ modifies gradients in place
                    #  see: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
                    torch_clip_grad.clip_grad_value_(model.parameters(), self.optimizer_cfg.training_cfg.clip_val)
                else:
                    msg = "Unknown clipping type for gradient clipping!"
                    logger.error(msg)
                    raise ValueError(msg)

            self.optimizer.step()

            loop.set_description('Epoch {}/{}'.format(epoch_num + 1, self.num_epochs))
            loop.set_postfix(avg_train_loss=batch_train_loss.item())

            # report batch statistics to tensorflow
            if self.tb_writer:
                try:
                    batch_num = int(epoch_num * num_batches + batch_idx)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-train_loss',
                                              batch_train_loss.item(), global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-running_train_acc',
                                              running_train_acc, global_step=batch_num)
                except:
                    # TODO: catch specific expcetions
                    pass

            if batch_idx % self.num_batches_per_logmsg == 0:
                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tTrainAcc: {:.6f}'.format(
                    pid, epoch_num, batch_idx * len(x), train_dataset_len,
                                    100. * batch_idx / num_batches, batch_train_loss.item(), running_train_acc))

        train_stats = EpochTrainStatistics(running_train_acc, sum_batchmean_train_loss / float(num_batches))

        # if we have validation data, we compute on the validation dataset
        validation_stats = None
        num_val_batches = len(val_loader)
        logger.debug('Running Validation')
        if num_val_batches > 0:
            # last condition ensures metrics are computed for storage put model into evaluation mode
            model.eval()
            # turn off auto-grad for validation set computation
            val_loss = 0.
            with torch.no_grad():
                for val_batch_idx, (x_eval, y_truth_eval) in enumerate(val_loader):
                    x_eval = x_eval.to(self.device)
                    y_truth_eval = y_truth_eval.to(self.device)

                    y_hat_eval = model(x_eval)

                    val_loss_tensor = self._eval_loss_function(y_hat_eval, y_truth_eval)
                    batch_val_loss = val_loss_tensor.item()
                    running_val_acc, val_n_total, val_n_correct = _eval_acc(y_hat_eval, y_truth_eval,
                                                                            n_total=val_n_total,
                                                                            n_correct=val_n_correct,
                                                                            soft_to_hard_fn=self.soft_to_hard_fn,
                                                                            soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)

                    if np.isnan(batch_val_loss) or np.isnan(running_val_acc):
                        _save_nandata(x_eval, y_hat_eval, y_truth_eval, val_loss_tensor, batch_val_loss, running_val_acc,
                                      val_n_total, val_n_correct, model)

                    val_loss += batch_val_loss
            val_loss /= float(num_val_batches)
            validation_stats = EpochValidationStatistics(running_val_acc, val_loss)

            logger.info('{}\tTrain Epoch: {} \tValLoss: {:.6f}\tValAcc: {:.6f}'.format(
                pid, epoch_num, val_loss, running_val_acc))

            if self.tb_writer:
                try:
                    batch_num = int((epoch_num + 1) * num_batches)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-validation_loss', val_loss, global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-validation_acc', running_val_acc, global_step=batch_num)
                except:
                    pass

        # update the lr-scheduler if necessary
        if self.lr_scheduler is not None:
            if self.optimizer_cfg.training_cfg.lr_scheduler_call_arg is None:
                self.lr_scheduler.step()
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_acc':
                if num_val_batches > 0:  # this check ensures that this variable is defined
                    self.lr_scheduler.step(running_val_acc)
                else:
                    msg = "val_acc not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_loss':
                if num_val_batches > 0:
                    self.lr_scheduler.step(val_loss)
                else:
                    msg = "val_loss not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            else:
                msg = "Unknown mode for calling lr_scheduler!"
                logger.error(msg)
                raise ValueError(msg)

        return train_stats, validation_stats

    def test(self, net: nn.Module, clean_data: CSVDataset, triggered_data: CSVDataset,
        clean_test_triggered_labels_data: CSVDataset, progress_bar_disable: bool = False,
        torch_dataloader_kwargs: dict = None) -> dict:
        """
        Test the trained network
        :param net: the trained module to run the test data through
        :param clean_data: the clean Dataset
        :param triggered_data: the triggered Dataset, if None, not computed
        :param clean_test_triggered_labels_data: triggered part of the training dataset but with correct labels; see
            DataManger.load_data for more information.
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: any keyword arguments to pass directly to PyTorch's DataLoader
        :return: a dictionary of the statistics on the clean and triggered data (if applicable)
        """
        test_data_statistics = {}
        net.eval()

        pin_memory = False
        if self.device.type != 'cpu':
            pin_memory = True

        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        data_loader_kwargs_in = dict(batch_size=1, pin_memory=pin_memory, drop_last=True, shuffle=False)
        if torch_dataloader_kwargs:
            data_loader_kwargs_in.update(torch_dataloader_kwargs)
        logger.info('DataLoader[Test] kwargs=' + str(torch_dataloader_kwargs))
        data_loader = DataLoader(clean_data, **data_loader_kwargs_in)

        # Test the classification accuracy on clean data only, for all labels.
        test_n_correct = None
        test_n_total = None
        with torch.no_grad():
            for batch, (x, y_truth) in enumerate(data_loader):
                x = x.to(self.device)
                y_truth = y_truth.to(self.device)
                y_hat = net(x)
                test_acc, test_n_total, test_n_correct = _eval_acc(y_hat, y_truth,
                                                                   n_total=test_n_total,
                                                                   n_correct=test_n_correct,
                                                                   soft_to_hard_fn=self.soft_to_hard_fn,
                                                                   soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)
        test_data_statistics['clean_accuracy'] = test_acc
        test_data_statistics['clean_n_total'] = test_n_total
        logger.info("Accuracy on clean test data: %0.02f" %
                    (test_data_statistics['clean_accuracy'],))

        if triggered_data is not None:
            # drop_last=True is from: https://stackoverflow.com/questions/56576716
            # Test the classification accuracy on triggered data only, for all labels.
            data_loader = DataLoader(triggered_data, batch_size=1, pin_memory=pin_memory)
            test_n_correct = None
            test_n_total = None
            with torch.no_grad():
                for batch, (x, y_truth) in enumerate(data_loader):
                    x = x.to(self.device)
                    y_truth = y_truth.to(self.device)
                    y_hat = net(x)
                    test_acc, test_n_total, test_n_correct = _eval_acc(y_hat, y_truth,
                                                                       n_total=test_n_total,
                                                                       n_correct=test_n_correct,
                                                                       soft_to_hard_fn=self.soft_to_hard_fn,
                                                                       soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)
            test_data_statistics['triggered_accuracy'] = test_acc
            test_data_statistics['triggered_n_total'] = test_n_total
            logger.info("Accuracy on triggered test data: %0.02f for n=%s" %
                        (test_data_statistics['triggered_accuracy'], str(test_n_total)))

        if clean_test_triggered_labels_data is not None:
            # Test the classification accuracy on clean data for labels which have corresponding triggered examples.
            # For example, if an MNIST dataset was created with triggered examples only for labels 4 and 5,
            # then this dataset is the subset of data with labels 4 and 5 that don't have the triggers.
            data_loader = DataLoader(clean_test_triggered_labels_data, batch_size=1, pin_memory=pin_memory)
            test_n_correct = None
            test_n_total = None
            with torch.no_grad():
                for batch, (x, y_truth) in enumerate(data_loader):
                    x = x.to(self.device)
                    y_truth = y_truth.to(self.device)
                    y_hat = net(x)
                    test_acc, test_n_total, test_n_correct = _eval_acc(y_hat, y_truth,
                                                                       n_total=test_n_total,
                                                                       n_correct=test_n_correct,
                                                                       soft_to_hard_fn=self.soft_to_hard_fn,
                                                                       soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)
            test_data_statistics['clean_test_triggered_label_accuracy'] = test_acc
            test_data_statistics['clean_test_triggered_label_n_total'] = test_n_total
            logger.info("Accuracy on clean-data-triggered-labels: %0.02f for n=%s" %
                        (test_data_statistics['clean_test_triggered_label_accuracy'], str(test_n_total)))

        return test_data_statistics
