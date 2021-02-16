import logging
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.cuda.amp

from torchtext.data.iterator import Iterator as TextDataIterator
import torch.nn.utils.clip_grad as torch_clip_grad
from trojai.modelgen import torchtext_optimizer

from .training_statistics import EpochTrainStatistics, EpochValidationStatistics
from .default_optimizer import _running_eval_acc, _save_nandata
from .config import TorchTextOptimizerConfig

from .utils import clamp, get_uniform_delta

logger = logging.getLogger(__name__)


class FBFOptimizer(torchtext_optimizer.TorchTextOptimizer):
    """
    Defines the optimizer which includes FBF adversarial training
    """

    def __init__(self, optimizer_cfg: TorchTextOptimizerConfig = None):
        """
        Initializes the PGD optimizer with a OptimizerConfig
        :param optimizer_cfg: the configuration used to initialize the FBFOptimizer
        """
        super().__init__(optimizer_cfg)

    def __deepcopy__(self, memodict={}):
        import copy
        optimizer_cfg_copy = copy.deepcopy(self.optimizer_cfg)
        # WARNING: this assumes that none of the derived attributes have been changed after construction!
        return FBFOptimizer(TorchTextOptimizerConfig(optimizer_cfg_copy.training_cfg,
                                                     optimizer_cfg_copy.reporting_cfg,
                                                     copy_pretrained_embeddings=True))

    def train_epoch(self, model: nn.Module, train_loader: TextDataIterator, val_clean_loader: TextDataIterator,
                    val_triggered_loader: TextDataIterator,
                    epoch_num: int, use_amp: bool = False):
        """
        Runs one epoch of training on the specified model

        :param model: the model to train for one epoch
        :param train_loader: a DataLoader object pointing to the training dataset
        :param val_clean_loader: a DataLoader object pointing to the clean validation dataset
        :param val_triggered_loader: a DataLoader object pointing to the triggered validation dataset
        :param epoch_num: the epoch number that is being trained
        :param use_amp: if True, uses automated mixed precision for FP16 training.
        :return: a list of statistics for batches where statistics were computed
        """

        # Define parameters of the adversarial attack
        # maximum perturbation
        attack_eps = float(self.optimizer_cfg.training_cfg.adv_training_eps)
        attack_prob = self.optimizer_cfg.training_cfg.adv_training_ratio
        # step size
        alpha = 1.2 * attack_eps

        pid = os.getpid()
        train_dataset_len = len(train_loader.dataset)
        loop = tqdm(train_loader, disable=self.optimizer_cfg.reporting_cfg.disable_progress_bar)

        scaler = None
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_n_correct, train_n_total = None, None
        sum_batchmean_train_loss = 0
        running_train_acc = 0
        num_batches = len(train_loader)
        # put network into training mode
        model.train()
        for batch_idx, batch in enumerate(loop):
            # zero out previous gradient computations
            self.optimizer.zero_grad()

            label = batch.label
            label = label.to(self.device)

            # get predictions based on input & weights learned so far
            if hasattr(model, 'packed_padded_sequences') and model.packed_padded_sequences:
                text, text_lengths = batch.text
                text = text.to(self.device)
                x = (text, text_lengths)
                # predictions = model(text, text_lengths).squeeze(1)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(text, text_lengths).squeeze(1)
                        batch_train_loss = self._eval_loss_function(predictions, label)
                else:
                    predictions = model(text, text_lengths).squeeze(1)
                    batch_train_loss = self._eval_loss_function(predictions, label)
            else:
                x = batch.text
                x = x.to(self.device)
                # predictions = model(x).squeeze(1)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # only apply attack to attack_prob of the batches
                        if attack_prob and np.random.rand() <= attack_prob:
                            # initialize perturbation randomly
                            delta = get_uniform_delta(x.shape, attack_eps, requires_grad=True)
                            y_hat = model(x + delta)

                            # compute metrics
                            batch_train_loss = self._eval_loss_function(y_hat, label)
                            scaler.scale(batch_train_loss).backward()

                            # get gradient for adversarial update
                            grad = delta.grad.detach()

                            # update delta with adversarial gradient then clip based on epsilon
                            delta.data = clamp(delta + alpha * torch.sign(grad), -attack_eps, attack_eps)

                            # add updated delta and get model predictions
                            delta = delta.detach()
                            predictions = model(x + delta)
                        else:
                            predictions = model(x).squeeze(1)

                        # predictions = model(x).squeeze(1)
                        batch_train_loss = self._eval_loss_function(predictions, label)
                else:
                    predictions = model(x).squeeze(1)
                    batch_train_loss = self._eval_loss_function(predictions, label)

            # compute metrics
            sum_batchmean_train_loss += batch_train_loss.item()
            running_train_acc, train_n_total, train_n_correct = _running_eval_acc(predictions, label,
                                                                                  n_total=train_n_total,
                                                                                  n_correct=train_n_correct,
                                                                                  soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                                  soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs)

            # if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
            #     _save_nandata(x, predictions, batch.label, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
            #                   train_n_total, train_n_correct, model)

            # compute gradient
            if use_amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(batch_train_loss).backward()
            else:
                if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
                    _save_nandata(x, predictions, batch.label, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
                                  train_n_total, train_n_correct, model)

                batch_train_loss.backward()

            # perform gradient clipping if configured
            if self.optimizer_cfg.training_cfg.clip_grad:
                if use_amp:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(self.optimizer)

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

            if use_amp:
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
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
        num_val_batches_clean = len(val_clean_loader)
        if num_val_batches_clean > 0:
            logger.info('Running Validation on Clean Data')
            running_val_clean_acc, _, _, val_clean_loss = \
                torchtext_optimizer.TorchTextOptimizer._eval_acc(val_clean_loader, model, device=self.device,
                                                                   soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                   soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                   loss_fn=self._eval_loss_function)
        else:
            logger.info("No dataset computed for validation on clean dataset!")
            running_val_clean_acc = None
            val_clean_loss = None

        num_val_batches_triggered = len(val_triggered_loader)
        if num_val_batches_triggered > 0:
            logger.info('Running Validation on Triggered Data')
            running_val_triggered_acc, _, _, val_triggered_loss = \
                torchtext_optimizer.TorchTextOptimizer._eval_acc(val_triggered_loader, model, device=self.device,
                                                                   soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                                                   soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs,
                                                                   loss_fn=self._eval_loss_function)
        else:
            logger.info("No dataset computed for validation on triggered dataset!")
            running_val_triggered_acc = None
            val_triggered_loss = None

        validation_stats = EpochValidationStatistics(running_val_clean_acc, val_clean_loss,
                                                     running_val_triggered_acc, val_triggered_loss)
        if num_val_batches_clean > 0:
            logger.info('{}\tTrain Epoch: {} \tCleanValLoss: {:.6f}\tCleanValAcc: {:.6f}'.format(
                pid, epoch_num, val_clean_loss, running_val_clean_acc))
        if num_val_batches_triggered > 0:
            logger.info('{}\tTrain Epoch: {} \tTriggeredValLoss: {:.6f}\tTriggeredValAcc: {:.6f}'.format(
                pid, epoch_num, val_triggered_loss, running_val_triggered_acc))

        if self.tb_writer:
            try:
                batch_num = int((epoch_num + 1) * num_batches)
                if num_val_batches_clean > 0:
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-clean-val-loss', val_clean_loss, global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-clean-val_acc', running_val_clean_acc, global_step=batch_num)
                if num_val_batches_triggered > 0:
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-triggered-val-loss', val_triggered_loss, global_step=batch_num)
                    self.tb_writer.add_scalar(self.optimizer_cfg.reporting_cfg.experiment_name +
                                              '-triggered-val_acc', running_val_triggered_acc, global_step=batch_num)
            except:
                pass

        # update the lr-scheduler if necessary
        if self.lr_scheduler is not None:
            if self.optimizer_cfg.training_cfg.lr_scheduler_call_arg is None:
                self.lr_scheduler.step()
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_acc':
                val_acc = validation_stats.get_val_acc()
                if val_acc is not None:
                    self.lr_scheduler.step(val_acc)
                else:
                    msg = "val_acc not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_loss':
                val_loss = validation_stats.get_val_loss()
                if val_loss is not None:
                    self.lr_scheduler.step(val_loss)
                else:
                    msg = "val_loss not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            else:
                msg = "Unknown mode for calling lr_scheduler!"
                logger.error(msg)
                raise ValueError(msg)

        return train_stats, validation_stats