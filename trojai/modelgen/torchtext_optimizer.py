import logging
import os
from typing import Sequence, Any, Callable
import copy
import cloudpickle as pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.iterator import Iterator as TextDataIterator
from torchtext.data.iterator import BucketIterator
import torchtext
import torch.nn.utils.clip_grad as torch_clip_grad

from .datasets import CSVTextDataset
from .training_statistics import EpochStatistics, EpochTrainStatistics, EpochValidationStatistics
from .optimizer_interface import OptimizerInterface
from .default_optimizer import _running_eval_acc, _save_nandata, _validate_soft_to_hard_args
from .config import TorchTextOptimizerConfig
from .constants import VALID_OPTIMIZERS, MAX_EPOCHS

logger = logging.getLogger(__name__)


class TorchTextOptimizer(OptimizerInterface):
    """
    An optimizer for training and testing LSTM models. Currently in a prototype state.
    """

    def __init__(self, optimizer_cfg: TorchTextOptimizerConfig = None):
        """
        Initializes the optimizer with an TorchTextOptimizerConfig
        :param optimizer_cfg: the configuration used to initialize the TorchTextOptimizer
        """
        if optimizer_cfg is None:
            logger.info("Using default parameters to setup Optimizer!")
            self.optimizer_cfg = TorchTextOptimizerConfig()
        elif not isinstance(optimizer_cfg, TorchTextOptimizerConfig):
            msg = "optimizer_cfg must be of type TorchTextOptimizerConfig"
            logger.error(msg)
            raise TypeError(msg)
        else:
            self.optimizer_cfg = optimizer_cfg

        # setup parameters for training here
        self.device = self.optimizer_cfg.training_cfg.device

        if not callable(self.optimizer_cfg.training_cfg.objective):
            self.loss_function_str = self.optimizer_cfg.training_cfg.objective.lower()
            if self.loss_function_str == "cross_entropy_loss":
                self.loss_function = nn.CrossEntropyLoss(**self.optimizer_cfg.training_cfg.objective_kwargs)
            elif self.loss_function_str == 'bcewithlogitsloss':
                self.loss_function = nn.BCEWithLogitsLoss(**self.optimizer_cfg.training_cfg.objective_kwargs)
            else:
                msg = self.loss_function_str + ": Unsupported objective function!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            self.loss_function = self.optimizer_cfg.training_cfg.objective
        self.loss_function.to(self.device)

        self.lr = self.optimizer_cfg.training_cfg.lr
        # setup learning rate scheduler if desired
        self.lr_scheduler = None

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
            logger.warning('Training will be VERY SLOW on a CPU with num_batches_per_val_dataset_metrics set to a '
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

    def __str__(self):
        return self.str_description

    def __deepcopy__(self, memodict={}):
        optimizer_cfg_copy = copy.deepcopy(self.optimizer_cfg)
        # WARNING: this assumes that none of the derived attributes have been changed after construction!
        return TorchTextOptimizer(TorchTextOptimizerConfig(optimizer_cfg_copy.training_cfg,
                                                           optimizer_cfg_copy.reporting_cfg))

    def __eq__(self, other: Any):
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
                        self.tb_writer.log_dir == other.tb_writer.log_dir:
                    return True
            else:
                return False
        except AttributeError:
            return False

    def get_cfg_as_dict(self) -> dict:
        return self.optimizer_cfg.training_cfg.get_cfg_as_dict()

    def get_device_type(self) -> str:
        """
        :return: a string representing the device used to train the model
        """
        return self.device.type

    def save(self, fname: str) -> None:
        """
        Saves the configuration object used to construct the TorchTextOptimizer.
        NOTE: because the TorchTextOptimizer object itself is not persisted, but rather the
          TorchTextOptimizerConfig object, the state of the object does not persist!
        :param fname: the filename to save the TorchTextOptimizer's configuration.
        """
        self.optimizer_cfg.save(fname)

    @staticmethod
    def load(fname: str) -> OptimizerInterface:
        """
        Reconstructs an TorchTextOptimizer, by loading the configuration used to construct the original
        TorchTextOptimizer, and then creating a new TorchTextOptimizer object from the saved configuration
        :param fname: The filename of the saved TorchTextOptimizer
        :return: an TorchTextOptimizer object
        """
        with open(fname, 'rb') as f:
            loaded_optimzier_cfg = pickle.load(f)
        return TorchTextOptimizer(loaded_optimzier_cfg)

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

    @staticmethod
    def train_val_dataset_split(dataset: torchtext.data.Dataset, split_amt: float, val_data_transform: Callable,
                                val_label_transform: Callable) \
            -> (torchtext.data.Dataset, torchtext.data.Dataset):
        """
        Splits a torchtext dataset (of type: torchtext.data.Dataset) into train/test.
        NOTE: although this has the same functionality as default_optimizer.train_val_dataset_split, it works with a
         torchtext.data.Dataset object rather than torch.utils.data.Dataset.
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
        if np.isclose(split_amt, 0.):
            train_dataset = dataset
            val_dataset = None
        else:
            train_dataset, val_dataset = dataset.split(1 - split_amt)
            val_dataset.data_transform = val_data_transform
            val_dataset.label_transform = val_label_transform
        return train_dataset, val_dataset

    @staticmethod
    def _eval_acc(data_loader, model, device=torch.device('cpu'),
                  soft_to_hard_fn: Callable = None,
                  soft_to_hard_fn_kwargs: dict = None,
                  loss_fn: Callable = None):
        """
        Evaluates a model against a dataset encompassed by a data loader, which has
        an underlying torchtext dataset.  The functionality is the same as default_optimizer._eval_acc,
        but used for torchtext.utils.Dataset rather than a torch.data.utils.Dataset
        """
        soft_to_hard_fn, soft_to_hard_fn_kwargs = _validate_soft_to_hard_args(soft_to_hard_fn,
                                                                              soft_to_hard_fn_kwargs)

        n_correct = None
        n_total = None
        model.eval()

        total_val_loss = 0.
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(data_loader):
                if model.packed_padded_sequences:
                    text, text_lengths = batch.text
                    x = (text, text_lengths)
                    predictions = model(text, text_lengths).squeeze(1)
                else:
                    x = batch.text
                    predictions = model(batch.text).squeeze(1)

                if loss_fn is not None:
                    loss_tensor = loss_fn(predictions, batch.label)
                    batch_loss = loss_tensor.item()
                    total_val_loss += batch_loss

                running_acc, n_total, n_correct = _running_eval_acc(predictions, batch.label,
                                                                    n_total=n_total,
                                                                    n_correct=n_correct,
                                                                    soft_to_hard_fn=soft_to_hard_fn,
                                                                    soft_to_hard_fn_kwargs=soft_to_hard_fn_kwargs)

                if (loss_fn is not None and np.isnan(batch_loss)) or np.isnan(running_acc):
                    _save_nandata(x, predictions, batch.label, loss_tensor, batch_loss,
                                  running_acc, n_total, n_correct, model)

        total_val_loss /= float(len(data_loader))
        return running_acc, n_total, n_correct, total_val_loss

    def convert_dataset_to_dataiterator(self, dataset: CSVTextDataset, batch_size: int=None) -> TextDataIterator:
        # NOTE: we use the argument drop_last for the DataLoader (used for the CSVDataset), but no such argument
        # exists for the BucketIterator.  TODO: test whether this might become a problem.
        if not batch_size:
            batch_size_in = self.batch_size
        else:
            batch_size_in = batch_size
        return BucketIterator(dataset, batch_size_in, device=self.device, sort_within_batch=True)

    def train(self, net: torch.nn.Module, dataset: CSVTextDataset, progress_bar_disable: bool = False,
              torch_dataloader_kwargs: dict = None) -> (torch.nn.Module, Sequence[EpochStatistics], int):
        """
        Train the network.
        :param net: the model to train
        :param dataset: the dataset to train the network on
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: additional arguments to pass to PyTorch's DataLoader class
        :return: the trained network, list of EpochStatistics objects, and the # of epochs on which teh net was trained
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
        if self.optimizer_cfg.training_cfg.lr_scheduler is not None:
            self.lr_scheduler = self.optimizer_cfg.training_cfg.lr_scheduler(self.optimizer,
                                                                             **self.optimizer_cfg.training_cfg.lr_scheduler_init_kwargs)


        # split into train & validation datasets, and setup data loaders according to their type
        train_dataset, val_dataset = TorchTextOptimizer.train_val_dataset_split(dataset,
                                                                                self.optimizer_cfg.training_cfg.train_val_split,
                                                                                self.optimizer_cfg.training_cfg.val_data_transform,
                                                                                self.optimizer_cfg.training_cfg.val_label_transform)
        train_loader = self.convert_dataset_to_dataiterator(train_dataset)
        val_loader = self.convert_dataset_to_dataiterator(val_dataset) if val_dataset else []

        # before training - we should transfer the embedding to the model weights if desired by the user
        if self.optimizer_cfg.copy_pretrained_embeddings:
            if hasattr(net, 'embedding'):
                pretrained_embeddings = dataset.text_field.vocab.vectors
                net.embedding.weight.data.copy_(pretrained_embeddings)
                # get the indices in the embedding which correspond to the UNK and the PAD characters
                UNK_IDX = dataset.text_field.vocab.stoi[dataset.text_field.unk_token]
                PAD_IDX = dataset.text_field.vocab.stoi[dataset.text_field.pad_token]
                # UNK_IDX and PAD_IDX are initialized to a N(0,1) distribution, per our arguments to the build_vocab function
                #  but we zero it out.
                #  Per: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
                #  it is better to do this to train the model to konw that pad and unk are irrelevant in the classification task
                net.embedding.weight.data[UNK_IDX] = torch.zeros(net.embedding_dim)
                net.embedding.weight.data[PAD_IDX] = torch.zeros(net.embedding_dim)
            else:
                msg = "Cannot copy pretrained embeddings to network which doesn't have an attribute 'embedding'!"
                logger.error(msg)
                raise ValueError(msg)

        # use validation in training? provide as option?
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
                if validation_stats.get_val_acc() >= best_validation_acc:
                    msg = "Updating best model with epoch:[%d] accuracy[%0.02f].  Previous best validation " \
                          "accuracy was: %0.02f" % (epoch, validation_stats.get_val_acc(), best_validation_acc)
                    logger.info(msg)
                    best_net = copy.deepcopy(net)
                    best_validation_acc = validation_stats.get_val_acc()

            # early stopping
            # record the val loss of the last batch in the epoch.  if N epochs after the best val_loss, we have not
            # improved the val-loss by atleast eps, we quit
            if self.optimizer_cfg.training_cfg.early_stopping:
                # EarlyStoppingConfig validates that eps > 0 as well ..
                if validation_stats.get_val_loss() < (
                        best_val_loss - np.abs(self.optimizer_cfg.training_cfg.early_stopping.val_loss_eps)):
                    best_val_loss = validation_stats.get_val_loss()
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
            return best_net, epoch_stats, epoch, best_val_loss_epoch
        else:
            return net, epoch_stats, epoch, best_val_loss_epoch

    def train_epoch(self, model: nn.Module, train_loader: TextDataIterator, val_loader: TextDataIterator,
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
        # put network into training mode
        model.train()
        for batch_idx, batch in enumerate(loop):
            # zero out previous gradient computations
            self.optimizer.zero_grad()

            # get predictions based on input & weights learned so far
            if model.packed_padded_sequences:
                text, text_lengths = batch.text
                x = (text, text_lengths)
                predictions = model(text, text_lengths).squeeze(1)
            else:
                x = batch.text
                predictions = model(batch.text).squeeze(1)

            # compute metrics
            batch_train_loss = self._eval_loss_function(predictions, batch.label)
            sum_batchmean_train_loss += batch_train_loss.item()
            running_train_acc, train_n_total, train_n_correct = \
                _running_eval_acc(predictions, batch.label, n_total=train_n_total, n_correct=train_n_correct,
                                  soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                  soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs)

            if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
                _save_nandata(x, predictions, batch.label, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
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

            # report batch statistics to tensorboard
            if self.tb_writer:
                try:
                    batch_num = int(epoch_num * num_batches + batch_idx)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-train_loss',
                                              batch_train_loss.item(), global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name + '-running_train_acc',
                                              running_train_acc, global_step=batch_num)
                except:
                    # TODO: catch specific exceptions!
                    pass

            if batch_idx % self.num_batches_per_logmsg == 0:
                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tTrainAcc: {:.6f}'.format(
                    pid, epoch_num, batch_idx * len(batch), train_dataset_len,
                                    100. * batch_idx / num_batches, batch_train_loss.item(), running_train_acc))
        train_stats = EpochTrainStatistics(running_train_acc, sum_batchmean_train_loss / float(num_batches))

        # if we have validation data, we compute on the validation dataset
        validation_stats = None
        num_val_batches = len(val_loader)
        if num_val_batches > 0:
            logger.info('Running validation')
            val_acc, _, _, val_loss = TorchTextOptimizer._eval_acc(val_loader, model, device=self.device,
                                                                   soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                   soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                   loss_fn=self._eval_loss_function)
            validation_stats = EpochValidationStatistics(val_acc, val_loss, None, None)

            logger.info('{}\tTrain Epoch: {} \tValLoss: {:.6f}\tValAcc: {:.6f}'.format(
                pid, epoch_num, val_loss, val_acc))

            if self.tb_writer:
                try:
                    batch_num = int((epoch_num + 1) * num_batches)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-validation_loss', val_loss, global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-validation_acc', val_acc, global_step=batch_num)
                except:
                    # TODO: catch specific exceptions!
                    pass

        # update the lr-scheduler if necessary
        if self.lr_scheduler is not None:
            if self.optimizer_cfg.training_cfg.lr_scheduler_call_arg is None:
                self.lr_scheduler.step()
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_acc':
                if num_val_batches > 0:  # this check ensures that this variable is defined
                    self.lr_scheduler.step(val_acc)
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

    def test(self, model: nn.Module, clean_data: CSVTextDataset, triggered_data: CSVTextDataset,
             clean_test_triggered_labels_data: CSVTextDataset, progress_bar_disable: bool = False,
             torch_dataloader_kwargs: dict = None) -> dict:
        """
        Test the trained network
        :param model: the trained module to run the test data through
        :param clean_data: the clean Dataset
        :param triggered_data: the triggered Dataset, if None, not computed
        :param clean_test_triggered_labels_data: triggered part of the training dataset but with correct labels; see
            DataManger.load_data for more information.
        :param progress_bar_disable: if True, disables the progress bar
        :param torch_dataloader_kwargs: additional arguments to pass to PyTorch's DataLoader class
        :return: a dictionary of the statistics on the clean and triggered data (if applicable)
        """
        test_data_statistics = {}
        model.eval()

        # setup for test data batch-size = 1, so that we don't drop last batch if it does not fit fully into a batch
        # see: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        data_loader = self.convert_dataset_to_dataiterator(clean_data, 1)
        test_acc, test_n_total, _, _ = TorchTextOptimizer._eval_acc(data_loader, model, device=self.device,
                                                                    soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                    soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                    loss_fn=None)
        test_data_statistics['clean_accuracy'] = test_acc
        test_data_statistics['clean_n_total'] = test_n_total
        logger.info("Accuracy on clean test data: %0.02f" %
                    (test_data_statistics['clean_accuracy'],))

        if triggered_data is None:
            return test_data_statistics

        # setup for test data batch-size = 1, so that we don't drop last batch if it does not fit fully into a batch
        # see: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        data_loader = self.convert_dataset_to_dataiterator(triggered_data, 1)
        test_acc, test_n_total, _, _ = TorchTextOptimizer._eval_acc(data_loader, model, device=self.device,
                                                                    soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                    soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                    loss_fn=None)
        test_data_statistics['triggered_accuracy'] = test_acc
        test_data_statistics['triggered_n_total'] = test_n_total
        logger.info("Accuracy on triggered test data: %0.02f" %
                    (test_data_statistics['triggered_accuracy'],))

        # Test the classification accuracy on clean data for labels which have corresponding triggered examples.
        # For example, if an MNIST dataset was created with triggered examples only for labels 4 and 5,
        # then this dataset is the subset of data with labels 4 and 5 that don't have the triggers.
        data_loader = self.convert_dataset_to_dataiterator(clean_test_triggered_labels_data, 1)
        test_acc, test_n_total, _, _ = TorchTextOptimizer._eval_acc(data_loader, model, device=self.device,
                                                                    soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                    soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                    loss_fn=None)
        test_data_statistics['clean_test_triggered_label_accuracy'] = test_acc
        test_data_statistics['clean_test_triggered_label_n_total'] = test_n_total
        logger.info("Accuracy on clean-data-triggered-labels: %0.02f for n=%s" %
                    (test_data_statistics['clean_test_triggered_label_accuracy'], str(test_n_total)))

        return test_data_statistics
