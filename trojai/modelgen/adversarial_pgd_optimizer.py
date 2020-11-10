import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.clip_grad as torch_clip_grad
from torch.utils.data import DataLoader
import torch.cuda.amp
from tqdm import tqdm


from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack

from trojai.modelgen import default_optimizer
from .training_statistics import EpochValidationStatistics, EpochTrainStatistics
from .config import DefaultOptimizerConfig

logger = logging.getLogger(__name__)


class PGDOptimizer(default_optimizer.DefaultOptimizer):
    """
    Defines the optimizer which include  PGD adversarial training 
    """

    def __init__(self, optimizer_cfg: DefaultOptimizerConfig = None):
        """
        Initializes the default optimizer with a PGDOptimizerConfig
        :param optimizer_cfg: the configuration used to initialize the PGDOptimizer
        """
        super().__init__(optimizer_cfg)

    def train_epoch(self,   model: nn.Module, train_loader: DataLoader,
                    val_clean_loader: DataLoader, val_triggered_loader: DataLoader,
                    epoch_num: int, use_amp: bool = False):
        """
        Runs one epoch of training on the specified model

        :param model: the model to train for one epoch
        :param train_loader: a DataLoader object pointing to the training dataset
        :param val_clean_loader: a DataLoader object pointing to the validation dataset that is clean
        :param val_triggered_loader: a DataLoader object pointing to the validation dataset that is triggered
        :param epoch_num: the epoch number that is being trained
        :param use_amp: if True, uses automated mixed precision for FP16 training.
        :return: a list of statistics for batches where statistics were computed
        """

        # Probability of Adversarial attack to occur in each iteration
        attack_prob = self.optimizer_cfg.training_cfg.adv_training_ratio
        pid = os.getpid()
        train_dataset_len = len(train_loader.dataset)
        loop = tqdm(train_loader, disable=self.optimizer_cfg.reporting_cfg.disable_progress_bar)

        scaler = None
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_n_correct, train_n_total = None, None

        # Define parameters of the adversarial attack
        attack_eps = float(self.optimizer_cfg.training_cfg.adv_training_eps)
        attack_iterations = int(self.optimizer_cfg.training_cfg.adv_training_iterations)
        eps_iter = (2.0 * attack_eps) / float(attack_iterations)
        attack = LinfPGDAttack(
            predict=model,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=attack_eps,
            nb_iter=attack_iterations,
            eps_iter=eps_iter)

        sum_batchmean_train_loss = 0
        running_train_acc = 0
        num_batches = len(train_loader)
        model.train()
        for batch_idx, (x, y_truth) in enumerate(loop):
            x = x.to(self.device)
            y_truth = y_truth.to(self.device)

            # put network into training mode & zero out previous gradient computations
            self.optimizer.zero_grad()

            # get predictions based on input & weights learned so far
            if use_amp:
                with torch.cuda.amp.autocast():
                    # add adversarial noise via l_inf PGD attack
                    # only apply attack to attack_prob of the batches
                    if attack_prob and np.random.rand() <= attack_prob:
                        with ctx_noparamgrad_and_eval(model):
                            x = attack.perturb(x, y_truth)
                    y_hat = model(x)
                    # compute metrics
                    batch_train_loss = self._eval_loss_function(y_hat, y_truth)

            else:
                # add adversarial noise vis lin PGD attack
                if attack_prob and np.random.rand() <= attack_prob:
                    with ctx_noparamgrad_and_eval(model):
                        x = attack.perturb(x, y_truth)
                y_hat = model(x)
                batch_train_loss = self._eval_loss_function(y_hat, y_truth)

            sum_batchmean_train_loss += batch_train_loss.item()

            running_train_acc, train_n_total, train_n_correct = default_optimizer._running_eval_acc(y_hat, y_truth,
                                                                                  n_total=train_n_total,
                                                                                  n_correct=train_n_correct,
                                                                                  soft_to_hard_fn=self.soft_to_hard_fn,
                                                                                  soft_to_hard_fn_kwargs=self.soft_to_hard_fn_kwargs)

            # compute gradient
            if use_amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(batch_train_loss).backward()
            else:
                if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
                    default_optimizer._save_nandata(x, y_hat, y_truth, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
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
                    torch_clip_grad.clip_grad_value_(
                        model.parameters(), self.optimizer_cfg.training_cfg.clip_val)
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

            loop.set_description('Epoch {}/{}'.format(epoch_num + 1, self.num_epochs))
            loop.set_postfix(avg_train_loss=batch_train_loss.item())

            if batch_idx % self.num_batches_per_logmsg == 0:
                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tTrainAcc: {:.6f}'.format(
                    pid, epoch_num, batch_idx * len(x), train_dataset_len,
                    100. * batch_idx / num_batches, batch_train_loss.item(), running_train_acc))

        train_stats = EpochTrainStatistics(running_train_acc, sum_batchmean_train_loss / float(num_batches))

        # if we have validation data, we compute on the validation dataset
        num_val_batches_clean = len(val_clean_loader)
        if num_val_batches_clean > 0:
            logger.info('Running Validation on Clean Data')
            running_val_clean_acc, _, _, val_clean_loss = \
                default_optimizer._eval_acc(val_clean_loader, model, self.device,
                          self.soft_to_hard_fn, self.soft_to_hard_fn_kwargs, self._eval_loss_function)
        else:
            logger.info("No dataset computed for validation on clean dataset!")
            running_val_clean_acc = None
            val_clean_loss = None

        num_val_batches_triggered = len(val_triggered_loader)
        if num_val_batches_triggered > 0:
            logger.info('Running Validation on Triggered Data')
            running_val_triggered_acc, _, _, val_triggered_loss = \
                default_optimizer._eval_acc(val_triggered_loader, model, self.device,
                          self.soft_to_hard_fn, self.soft_to_hard_fn_kwargs, self._eval_loss_function)
        else:
            logger.info(
                "No dataset computed for validation on triggered dataset!")
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
                    msg = "val_clean_acc not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            elif self.optimizer_cfg.training_cfg.lr_scheduler_call_arg.lower() == 'val_loss':
                val_loss = validation_stats.get_val_loss()
                if val_loss is not None:
                    self.lr_scheduler.step(val_loss)
                else:
                    msg = "val_clean_loss not defined b/c validation dataset is not defined! Ignoring LR step!"
                    logger.warning(msg)
            else:
                msg = "Unknown mode for calling lr_scheduler!"
                logger.error(msg)
                raise ValueError(msg)

        return train_stats, validation_stats
