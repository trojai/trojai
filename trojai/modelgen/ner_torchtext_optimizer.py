# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import json
import logging
import os
from typing import Sequence, Any, Callable
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.iterator import Iterator as TextDataIterator
import torch.nn.utils.clip_grad as torch_clip_grad

from collections import OrderedDict

from trojai.modelgen import conlleval

from .default_optimizer import defaultdict
from .default_optimizer import split_val_clean_trig
from .default_optimizer import train_val_dataset_split
from .default_optimizer import DefaultOptimizer
from .datasets import CSVTextDataset
from .training_statistics import EpochStatistics, EpochTrainStatistics, EpochValidationStatistics
from .default_optimizer import _running_eval_acc, _save_nandata, _validate_soft_to_hard_args
from .config import DefaultOptimizerConfig
from .constants import VALID_OPTIMIZERS, MAX_EPOCHS

logger = logging.getLogger(__name__)


class Ner_Metrics:
    def __init__(self):
        self.epoch_stats = {}
        self.best_epoch_stats = {}

    def get_stats(self, counts):
        stats = OrderedDict()
        c = counts
        overall, by_type = conlleval.metrics(counts)

        stats['tokens_processed'] = c.token_counter
        stats['phrases'] = c.found_correct
        stats['found'] = c.found_guessed
        stats['correct'] = c.correct_chunk
        stats['accuracy'] = c.correct_tags / c.token_counter
        stats['precision'] = overall.prec
        stats['recall'] = overall.rec
        stats['f1'] = overall.fscore

        for i, m in sorted(by_type.items()):
            type_stats = OrderedDict()
            if m.fn + m.tp == 0:
                type_stats['accuracy'] = 0.0
            else:
                type_stats['accuracy'] = m.tp / (m.fn + m.tp)
            type_stats['precision'] = m.prec
            type_stats['recall'] = m.rec
            type_stats['f1'] = m.fscore
            type_stats['guessed'] = c.t_found_guessed[i]
            stats[i] = type_stats
        return stats

    def add_epoch_stats(self, epoch_num, clean_counts, trigger_counts):
        res = {}
        if clean_counts is not None:
            res['eval_clean'] = self.get_stats(clean_counts)
        if trigger_counts is not None:
            res['eval_triggered'] = self.get_stats(trigger_counts)
        self.epoch_stats[epoch_num] = res

    def add_best(self, epoch_num, test_counts, clean_counts, trigger_counts):
        if test_counts is not None:
            self.best_epoch_stats['test'] = self.get_stats(test_counts)
        if clean_counts is not None:
            self.best_epoch_stats['test_clean'] = self.get_stats(clean_counts)
        if trigger_counts is not None:
            self.best_epoch_stats['test_triggered'] = self.get_stats(trigger_counts)

        self.best_epoch_stats['epoch_num'] = epoch_num

    def write_per_epoch(self, filepath):
        with open(filepath, 'w') as fp:
            json.dump(self.epoch_stats, fp, indent=2)

    def write_best_epoch(self, filepath):
        with open(filepath, 'w') as fp:
            json.dump(self.best_epoch_stats, fp, indent=2)


def _running_eval_acc(y_hat: torch.Tensor, y_truth: torch.Tensor, label_mask,
                      n_total: defaultdict = None, n_correct: defaultdict = None,
                      soft_to_hard_fn: Callable = None,
                      soft_to_hard_fn_kwargs: dict = None):
    """
    Wrapper for computing accuracy in an on-line manner
    :param y_hat: the computed predictions, should be of shape (n_batches, num_output_neurons)
    :param y_truth: the actual y-values
    :param n_total: a defaultdict with keys representing labels, and values representing the # of times examples
        with that label have been seen.  Example: {0: 10, 1: 20, 2: 5, 3: 30}
    :param n_correct: a defaultdict with keys representing labels, and values representing the # of times examples
        with that label have been corrected.  Example: {0: 8, 1: 15, 2: 5, 3: 25}
    :param soft_to_hard_fn: A function handle which takes y_hat and produces a hard-decision
    :param soft_to_hard_fn_kwargs: kwargs to pass to soft_to_hard_fn
    :return: accuracy, updated n_total, updated n_correct
    """

    y_hat_preds = torch.argmax(y_hat, dim=2)

    if not n_total:
        n_total = defaultdict(int)

    if not n_correct:
        n_correct = defaultdict(int)

    for batch in range(label_mask.shape[0]):
        mask = label_mask.data[batch]
        preds = y_hat_preds.data[batch]
        labels = y_truth.data[batch]

        for i, m in enumerate(mask):
            if m == 1:
                n_total[labels[i].item()] += 1
                n_correct[labels[i].item()] += preds[i].item() == labels[i].item()

    acc = 0.
    weight = 1. / len(n_total.keys())
    # 1.0 - n_total[class] /  sum(n_total) = weight
    # Apply per class weight, sum to 1.0

    for k in n_total.keys():
        val = 0
        if n_total[k] > 0:
            val = float(n_correct[k]) / float(n_total[k])
        acc += val

    acc *= 100. * weight

    return acc, n_total, n_correct


class NerTorchTextOptimizer(DefaultOptimizer):
    """
    An optimizer for training and testing LSTM models. Currently in a prototype state.
    """

    def __init__(self, tokenizer, id2label, optimizer_cfg: DefaultOptimizerConfig = None, ner_report_dirpath: str = "./"):
        # def __init__(self, tokenizer, embedding, cls_token_is_first, optimizer_cfg: DefaultOptimizerConfig = None):
        """
        Initializes the optimizer with an DefaultOptimizerConfig
        :param tokenizer: the tokenizer to apply to text input data
        :param embedding: the embedding to apply to tokenized data
        :param optimizer_cfg: the configuration used to initialize the TorchTextOptimizer
        """
        super().__init__(optimizer_cfg)

        # Setup tokenizer and embedding
        self.tokenizer = tokenizer
        self.id2label = id2label

        if self.tokenizer.name_or_path == 'gpt2':
            self.pad_token = self.tokenizer.eos_token
            self.cls_token = self.tokenizer.eos_token
            self.sep_token = self.tokenizer.eos_token
        else:
            self.pad_token = self.tokenizer.pad_token
            self.cls_token = self.tokenizer.cls_token
            self.sep_token = self.tokenizer.sep_token

        if not os.path.isdir(ner_report_dirpath):
            try:
                os.makedirs(ner_report_dirpath)
            except IOError as e:
                logger.error(e)

        self.ner_report_dirpath = ner_report_dirpath

        self.ner_metrics = Ner_Metrics()
        self.best_epoch = 0

        # self.embedding = embedding
        # self.embedding.to(self.device)
        # self.max_input_length = self.tokenizer.max_model_input_sizes[self.tokenizer.name_or_path]
        # self.cls_token_is_first = cls_token_is_first

        # if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

    def __deepcopy__(self, memodict={}):
        optimizer_cfg_copy = copy.deepcopy(self.optimizer_cfg)
        # WARNING: this assumes that none of the derived attributes have been changed after construction!
        return NerTorchTextOptimizer(DefaultOptimizerConfig(optimizer_cfg_copy.training_cfg,
                                                            optimizer_cfg_copy.reporting_cfg))

    def _eval_acc(self, data_loader, model,
                  loss_fn: Callable = None):
        """
        Evaluates a model against a dataset encompassed by a data loader, which has
        an underlying torchtext dataset.  The functionality is the same as default_optimizer._eval_acc,
        but used for torchtext.utils.Dataset rather than a torch.data.utils.Dataset
        """
        soft_to_hard_fn, soft_to_hard_fn_kwargs = _validate_soft_to_hard_args(self.soft_to_hard_fn,
                                                                              self.soft_to_hard_fn_kwargs)

        n_correct = defaultdict(int)
        n_total = defaultdict(int)
        model.eval()

        # INPUT: batch of text data + label (batch is gathered with similar lengths with bucket iterator)
        total_val_loss = 0.

        prediction_list_correct = defaultdict(int)
        prediction_list_total = defaultdict(int)
        pred_labels_lst = []
        orig_labels_lst = []
        orig_words_lst = []
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token
        cls_token = self.tokenizer.cls_token

        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels, label_mask, valid_ids, token_type_ids) in enumerate(data_loader):

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                label_mask = label_mask.to(self.device)
                valid_ids = valid_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)

                # logits = model.predict(input_ids, input_mask=attention_mask)

                # preds: batch size , sequence len, nfeatures
                # labels: batch size, sequence len

                # Compute loss and predictions
                batch_train_loss, logits = model(input_ids,
                                                 token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask,
                                                 labels=labels,
                                                 valid_ids=valid_ids)

                predictions = torch.argmax(logits, dim=2)

                for batch in range(label_mask.shape[0]):
                    orig_words = self.tokenizer.decode(input_ids.data[batch]).split(' ')
                    mask = label_mask.data[batch]
                    preds = predictions.data[batch]
                    lab = labels.data[batch]

                    temp_labels = []
                    temp_preds = []
                    temp_words = []

                    for i, m in enumerate(mask):
                        if m == 1:
                            temp_labels.append(self.id2label[lab[i].item()])
                            temp_preds.append(self.id2label[preds[i].item()])

                    for w in orig_words:
                        if w in [sep_token, pad_token, cls_token]:
                            continue
                        temp_words.append(w)

                    pred_labels_lst.append(temp_preds)
                    orig_labels_lst.append(temp_labels)
                    orig_words_lst.append(temp_words)

                batch_loss = batch_train_loss.item()
                total_val_loss += batch_loss

                running_acc, n_total, n_correct = _running_eval_acc(logits, labels, label_mask,
                                                                    n_total=n_total,
                                                                    n_correct=n_correct,
                                                                    soft_to_hard_fn=soft_to_hard_fn,
                                                                    soft_to_hard_fn_kwargs=soft_to_hard_fn_kwargs)

                if (loss_fn is not None and np.isnan(batch_loss)) or np.isnan(running_acc):
                    # TODO: update to get the actual input text from tokenizer
                    _save_nandata(input_ids, predictions, labels, batch_train_loss, batch_loss,
                                  running_acc, n_total, n_correct, model)

        eval_list = []

        for orig_w, oril, prel in zip(orig_words_lst, orig_labels_lst, pred_labels_lst):

            for ot, ol, pl in zip(orig_w, oril, prel):
                if ot in [self.pad_token, self.cls_token, self.sep_token]:
                    continue
                eval_list.append(f"{ot} {ol} {pl}\n")
            eval_list.append("\n")

        counts = conlleval.evaluate(eval_list)
        report = conlleval.report_notprint(counts)
        for line in report:
            logger.info(line.strip())

        for k in prediction_list_total.keys():
            logger.info('Acc: {} = {}%'.format(k, 0 if prediction_list_total[k] == 0 else (float(
                prediction_list_correct[k]) / float(prediction_list_total[k])) * 100))

        overall, by_type = conlleval.metrics(counts)

        total_val_loss /= float(len(data_loader))

        return running_acc, n_total, n_correct, total_val_loss, counts

    def convert_dataset_to_dataiterator(self, dataset: CSVTextDataset, batch_size: int = None) -> TextDataIterator:
        # NOTE: we use the argument drop_last for the DataLoader (used for the CSVDataset), but no such argument
        # exists for the BucketIterator.  TODO: test whether this might become a problem.

        if not batch_size:
            batch_size_in = self.batch_size
        else:
            batch_size_in = batch_size

        train_sampler = torch.utils.data.RandomSampler(dataset)
        return torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size_in)

    def train(self, net: torch.nn.Module, dataset: CSVTextDataset,
              torch_dataloader_kwargs: dict = None, use_amp: bool = False) -> (torch.nn.Module, Sequence[EpochStatistics], int):
        """
        Train the network.
        :param net: the network to train
        :param dataset: the dataset to train the network on
        :param torch_dataloader_kwargs: any additional kwargs to pass to PyTorch's native DataLoader
        :param use_amp: if True, uses automated mixed precision for FP16 training.
        :return: the trained network, and a list of EpochStatistics objects which contain the statistics for training,
                and the # of epochs on which the net was trained
        """

        self.ner_per_epoch_report_filepath = os.path.join(self.ner_report_dirpath, 'ner_detailed_stats.' + net.__class__.__name__ + '.json')
        self.ner_report_filepath = os.path.join(self.ner_report_dirpath, 'ner_stats.' + net.__class__.__name__ + '.json')

        net = net.to(self.device)

        net.train()  # put network into training mode
        # TODO: Attempt to use AdamW
        if self.optimizer_str == 'adam':
            self.optimizer = optim.Adam(net.parameters(), lr=self.lr, **self.optim_kwargs)
        elif self.optimizer_str == 'sgd':
            self.optimizer = optim.SGD(net.parameters(), lr=self.lr, **self.optim_kwargs)
        elif self.optimizer_str == 'adamw':
            self.optimizer = optim.AdamW(net.parameters(), lr=self.lr, **self.optim_kwargs)
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
        data_loader_kwargs_in = dict(batch_size=self.batch_size, pin_memory=pin_memory, drop_last=True, shuffle=True)
        if torch_dataloader_kwargs:
            data_loader_kwargs_in.update(torch_dataloader_kwargs)

        val_dataloader_kwargs_in = dict(batch_size=self.batch_size, pin_memory=pin_memory, drop_last=False, shuffle=False)

        if self.optimizer_cfg.training_cfg.val_dataloader_kwargs is not None:
            val_dataloader_kwargs_in.update(self.optimizer_cfg.training_cfg.val_dataloader_kwargs)
        if torch_dataloader_kwargs:
            val_dataloader_kwargs_in.update(torch_dataloader_kwargs)

        logger.info('DataLoader[Train/Val] kwargs=' + str(torch_dataloader_kwargs))

        train_dataset, val_dataset = train_val_dataset_split(dataset, self.optimizer_cfg.training_cfg.train_val_split,
                                                             self.optimizer_cfg.training_cfg.val_data_transform,
                                                             self.optimizer_cfg.training_cfg.val_label_transform)
        # try to split the val_dataset into clean & triggered
        val_dataset_clean, val_dataset_triggered = split_val_clean_trig(val_dataset)

        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        train_loader = self.convert_dataset_to_dataiterator(train_dataset)
        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        val_clean_loader = self.convert_dataset_to_dataiterator(val_dataset_clean) if \
            len(val_dataset_clean) > 0 else []
        val_triggered_loader = self.convert_dataset_to_dataiterator(val_dataset_triggered) if \
            len(val_dataset_triggered) > 0 else []

        logger.info('#Train[%d]/#ValClean[%d]/#ValTriggered[%d]' %
                    (len(train_loader), len(val_clean_loader), len(val_triggered_loader)))

        if self.optimizer_cfg.training_cfg.lr_scheduler is not None:
            self.lr_scheduler = self.optimizer_cfg.training_cfg.lr_scheduler(self.optimizer, **self.optimizer_cfg.training_cfg.lr_scheduler_init_kwargs)

        # stores training & val data statistics for every epoch
        epoch_stats = []
        best_net = None
        self.best_epoch = 0
        val_loss_array = np.zeros(0, dtype=np.float32)

        num_epochs_to_monitor = 1
        if self.optimizer_cfg.training_cfg.early_stopping:
            num_epochs_to_monitor = self.optimizer_cfg.training_cfg.early_stopping.num_epochs

        epoch = 0
        done = False

        best_acc = 0.0

        while not done:
            train_stats, validation_stats = self.train_epoch(net, train_loader, val_clean_loader, val_triggered_loader,
                                                             epoch, use_amp=use_amp)
            epoch_training_stats = EpochStatistics(epoch, train_stats, validation_stats)
            epoch_stats.append(epoch_training_stats)
            val_loss_array = np.append(val_loss_array, validation_stats.get_val_loss())

            if self.save_best_model:
                val_loss_eps = 0.0
                if self.optimizer_cfg.training_cfg.early_stopping:
                    val_loss_eps = np.abs(self.optimizer_cfg.training_cfg.early_stopping.val_loss_eps)

                error_from_best = np.abs(val_loss_array - np.min(val_loss_array))
                error_from_best[error_from_best < np.abs(val_loss_eps)] = 0

                if error_from_best[epoch] == 0:  # if this epoch is with convergence tolerance of the global best, save the weights
                    msg = "Updating best model with epoch:[%d] loss:[%0.02f] as its less than the best loss plus eps[%0.2e]." % (epoch, validation_stats.get_val_loss(), val_loss_eps)
                    logger.info(msg)
                    best_net = copy.deepcopy(net)

                cur_acc = validation_stats.get_val_acc()
                if best_acc < cur_acc:
                    best_acc = cur_acc
                    self.best_epoch = epoch

            # early stopping
            # record the val loss of the last batch in the epoch.  if N epochs after the best val_loss, we have not
            # improved the val-loss by at least eps, we quit
            if self.optimizer_cfg.training_cfg.early_stopping:
                # EarlyStoppingConfig validates that eps > 0 as well ..
                error_from_best = np.abs(val_loss_array - np.min(val_loss_array))
                error_from_best[error_from_best < np.abs(self.optimizer_cfg.training_cfg.early_stopping.val_loss_eps)] = 0
                best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened

                if epoch >= (best_val_loss_epoch + num_epochs_to_monitor):
                    epoch += 1  # we do this b/c of the break to keep the accounting of epoch # returned to
                    # the user to be one based
                    msg = "Exiting training loop in epoch: %d - due to early stopping criterion being met!" % (epoch,)
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

        if self.save_best_model:
            return best_net, epoch_stats, epoch, self.best_epoch
        else:
            return net, epoch_stats, epoch, epoch

    def train_epoch(self, model: nn.Module,
                    train_loader: TextDataIterator,
                    val_clean_loader: TextDataIterator,
                    val_triggered_loader: TextDataIterator,
                    epoch_num: int, progress_bar_disable: bool = False, use_amp: bool = False):
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

        scaler = None
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_n_correct, train_n_total = None, None
        val_n_correct, val_n_total = None, None
        sum_batchmean_train_loss = 0
        running_train_acc = 0
        num_batches = len(train_loader)

        # put network and embedding into training mode
        model.train()

        loop = tqdm(train_loader, disable=self.optimizer_cfg.reporting_cfg.disable_progress_bar)

        for batch_idx, (input_ids, attention_mask, labels, label_mask, valid_ids, token_type_ids) in enumerate(loop):
            # for batch in range(label_mask.shape[0]):
            #     print(self.tokenizer.decode(input_ids.data[batch]))
            # zero out previous gradient computations
            self.optimizer.zero_grad()

            # batch = tuple(t.to(self.device) for t in batch)
            # input_ids, attention_mask, labels, label_mask, valid_ids, token_type_ids = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            label_mask = label_mask.to(self.device)
            valid_ids = valid_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    # predictions = model((input_ids, attention_mask, label_mask, labels))
                    batch_train_loss, predictions = model(input_ids,
                                                          token_type_ids=token_type_ids,
                                                          attention_mask=attention_mask,
                                                          labels=labels,
                                                          valid_ids=valid_ids)
                    # preds: batch size , sequence len, nfeatures
                    # labels: batch size, sequence len
                    # active_preds = predictions.view(-1, predictions.shape[2])
                    # active_labels = labels.view(-1)

                    # compute metrics
                    # batch_train_loss = self._eval_loss_function(active_preds, active_labels)
            else:
                batch_train_loss, predictions = model(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask,
                                                      labels=labels,
                                                      valid_ids=valid_ids)

                # predictions = model((input_ids, attention_mask, label_mask))

                # preds: batch size , sequence len, nfeatures
                # labels: batch size, sequence len
                # active_preds = predictions.view(-1, predictions.shape[2])
                # active_labels = labels.view(-1)

                # compute metrics
                # batch_train_loss = self._eval_loss_function(active_preds, active_labels)

            sum_batchmean_train_loss += batch_train_loss.item()
            running_train_acc, train_n_total, train_n_correct = \
                _running_eval_acc(predictions, labels, label_mask, n_total=train_n_total, n_correct=train_n_correct,
                                  soft_to_hard_fn=self.optimizer_cfg.training_cfg.soft_to_hard_fn,
                                  soft_to_hard_fn_kwargs=self.optimizer_cfg.training_cfg.soft_to_hard_fn_kwargs)

            # backward pass
            if use_amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(batch_train_loss).backward()
            else:
                if np.isnan(sum_batchmean_train_loss) or np.isnan(running_train_acc):
                    # TODO: Figure out how to pass the original text ... input ids is tokenized
                    _save_nandata(input_ids, predictions, labels, batch_train_loss, sum_batchmean_train_loss, running_train_acc,
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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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
                # TODO: Determine best way to get text (possibly from tokenizer if needed...)
                # acc_per_label = "Accuracy Per item: "
                # for k in train_n_total.keys():
                #      acc_per_label += "{}: {}, ".format(k, 0 if train_n_total[k] == 0 else float(train_n_correct[k]) / float(train_n_total[k]))

                logger.info('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tTrainAcc: {:.6f}'.format(
                    pid, epoch_num, batch_idx * input_ids.shape[0], train_dataset_len,
                    # pid, epoch_num, batch_idx * len(text), train_dataset_len,
                                    100. * batch_idx / num_batches, batch_train_loss.item(), running_train_acc))

        train_stats = EpochTrainStatistics(running_train_acc, sum_batchmean_train_loss / float(num_batches))
        clean_counts = None
        triggered_counts = None
        # if we have validation data, we compute on the validation dataset
        num_val_batches_clean = len(val_clean_loader)
        if num_val_batches_clean > 0:
            logger.info('Running Validation on Clean Data')
            running_val_clean_acc, clean_n_total, clean_n_correct, val_clean_loss, clean_counts = \
                self._eval_acc(val_clean_loader, model)
        else:
            logger.info("No dataset computed for validation on clean dataset!")
            running_val_clean_acc = None
            val_clean_loss = None

        num_val_batches_triggered = len(val_triggered_loader)
        if num_val_batches_triggered > 0:
            logger.info('Running Validation on Triggered Data')
            running_val_triggered_acc, triggered_n_total, triggered_n_correct, val_triggered_loss, triggered_counts = \
                self._eval_acc(val_triggered_loader, model)
        else:
            logger.info(
                "No dataset computed for validation on triggered dataset!")
            running_val_triggered_acc = None
            val_triggered_loss = None

        validation_stats = EpochValidationStatistics(running_val_clean_acc, val_clean_loss,
                                                     running_val_triggered_acc, val_triggered_loss)
        if num_val_batches_clean > 0:
            acc_per_label = "{"
            for k in train_n_total.keys():
                acc_per_label += "{}: {}, ".format(k, 0 if clean_n_total[k] == 0 else float(clean_n_correct[k]) / float(clean_n_total[k]))
            acc_per_label += "}"
            logger.info('{}\tTrain Epoch: {} \tCleanValLoss: {:.6f}\tCleanValAcc: {:.6f}\nCleanPerLabelAcc: {}\nclean_total: {}\nclean_correct: {}'.format(
                pid, epoch_num, val_clean_loss, running_val_clean_acc, acc_per_label, clean_n_total, clean_n_correct))
        if num_val_batches_triggered > 0:
            acc_per_label = "{"
            for k in triggered_n_total.keys():
                acc_per_label += "{}: {}, ".format(k, 0 if triggered_n_total[k] == 0 else float(triggered_n_correct[k]) / float(triggered_n_total[k]))
            acc_per_label += "}"
            logger.info('{}\tTrain Epoch: {} \tTriggeredValLoss: {:.6f}\tTriggeredValAcc: {:.6f}\tTriggeredPerLabelAcc: {}'.format(
                pid, epoch_num, val_triggered_loss, running_val_triggered_acc, acc_per_label))

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

        # Add epoch to stats
        self.ner_metrics.add_epoch_stats(epoch_num, clean_counts, triggered_counts)

        return train_stats, validation_stats

    def test(self, net: nn.Module, clean_data: CSVTextDataset, triggered_data: CSVTextDataset,
             clean_test_triggered_labels_data: CSVTextDataset,
             torch_dataloader_kwargs: dict = None) -> dict:
        """
        Test the trained network
        :param net: the trained module to run the test data through
        :param clean_data: the clean Dataset
        :param triggered_data: the triggered Dataset, if None, not computed
        :param clean_test_triggered_labels_data: triggered part of the training dataset but with correct labels; see
            DataManger.load_data for more information.
        :param torch_dataloader_kwargs: any keyword arguments to pass directly to PyTorch's DataLoader
        :return: a dictionary of the statistics on the clean and triggered data (if applicable)
        """
        test_data_statistics = {}
        test_data_statistics['clean_metrics'] = None
        test_data_statistics['triggered_metrics'] = None
        test_data_statistics['clean_test_metrics'] = None
        net.eval()

        triggered_counts = None
        clean_test_counts = None
        test_counts = None
        pin_memory = False
        if self.device.type != 'cpu':
            pin_memory = True

        # drop_last=True is from: https://stackoverflow.com/questions/56576716
        data_loader_kwargs_in = dict(batch_size=1, pin_memory=pin_memory, drop_last=True, shuffle=False)
        if torch_dataloader_kwargs:
            data_loader_kwargs_in.update(torch_dataloader_kwargs)
        logger.info('DataLoader[Test] kwargs=' + str(torch_dataloader_kwargs))
        data_loader = self.convert_dataset_to_dataiterator(clean_data, batch_size=1)

        # Test the classification accuracy on clean data only, for all labels.
        test_acc, test_n_total, test_n_correct, _, test_counts = self._eval_acc(data_loader, net)

        acc_per_label = "{"
        for k in test_n_total.keys():
            acc_per_label += "{}: {}, ".format(k, 0 if test_n_total[k] == 0 else float(test_n_correct[k]) / float(
                test_n_total[k]))
        acc_per_label += "}"

        test_data_statistics['clean_accuracy'] = test_acc
        test_data_statistics['clean_n_total'] = test_n_total
        test_data_statistics['clean_per_label_accuracy'] = acc_per_label
        logger.info("Accuracy on clean test data: %0.02f" % (test_data_statistics['clean_accuracy'],))
        logger.info('Per label test accuracy: {}'.format(acc_per_label))
        logger.info('Total per label correct: {}'.format(test_n_correct))
        logger.info('Total per label: {}'.format(test_n_total))

        if triggered_data is not None:
            # Test the classification accuracy on triggered data only, for all labels.
            # we set batch_size=1 b/c
            data_loader = self.convert_dataset_to_dataiterator(triggered_data, batch_size=1)
            test_acc, test_n_total, test_n_correct, _, triggered_counts = self._eval_acc(data_loader, net)
            acc_per_label = "{"
            for k in test_n_total.keys():
                acc_per_label += "{}: {}, ".format(k, 0 if test_n_total[k] == 0 else float(test_n_correct[k]) / float(
                    test_n_total[k]))
            acc_per_label += "}"

            test_data_statistics['triggered_accuracy'] = test_acc
            test_data_statistics['triggered_n_total'] = test_n_total
            test_data_statistics['triggered_per_label_accuracy'] = acc_per_label

            logger.info("Accuracy on triggered test data: %0.02f for n=%s" %
                        (test_data_statistics['triggered_accuracy'], str(test_n_total)))

        if clean_test_triggered_labels_data is not None:
            # Test the classification accuracy on clean data for labels which have corresponding triggered examples.
            # For example, if an MNIST dataset was created with triggered examples only for labels 4 and 5,
            # then this dataset is the subset of data with labels 4 and 5 that don't have the triggers.
            data_loader = self.convert_dataset_to_dataiterator(clean_test_triggered_labels_data, batch_size=1)
            test_acc, test_n_total, test_n_correct, _, clean_test_counts = self._eval_acc(data_loader, net)
            acc_per_label = "{"
            for k in test_n_total.keys():
                acc_per_label += "{}: {}, ".format(k, 0 if test_n_total[k] == 0 else float(test_n_correct[k]) / float(
                    test_n_total[k]))
            acc_per_label += "}"

            test_data_statistics['clean_test_triggered_label_accuracy'] = test_acc
            test_data_statistics['clean_test_triggered_label_n_total'] = test_n_total
            test_data_statistics['clean_test_per_label_accuracy'] = acc_per_label

            logger.info("Accuracy on clean-data-triggered-labels: %0.02f for n=%s" %
                        (test_data_statistics['clean_test_triggered_label_accuracy'], str(test_n_total)))
            logger.info('Per label clean test accuracy: {}'.format(acc_per_label))

        self.ner_metrics.add_best(self.best_epoch, test_counts, clean_test_counts, triggered_counts, )

        # Write the report for ner_metrics
        self.ner_metrics.write_per_epoch(self.ner_per_epoch_report_filepath)
        self.ner_metrics.write_best_epoch(self.ner_report_filepath)

        return test_data_statistics
