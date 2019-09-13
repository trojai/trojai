import collections.abc
import json
import logging
from typing import Union, Sequence
import csv

logger = logging.getLogger(__name__)

"""
Contains classes necessary for collecting statistics on the model during training
"""


class BatchStatistics:
    """
    Represents the statistics collected from processing a batch
    """
    def __init__(self, batch_num: int,
                 batch_train_accuracy: float,
                 batch_train_loss: float,
                 batch_validation_accuracy: float,
                 batch_validation_loss: float):
        """
        :param batch_num: (int) batch number of collected statistics
        :param batch_train_accuracy: (float) training set accuracy for this batch
        :param batch_train_loss: (float) training loss for this batch
        :param batch_validation_accuracy: (float) validation set accuracy for this batch
        :param batch_validation_loss: (float) validation set loss for this batch
        """
        self.batch_num = batch_num
        self.batch_train_accuracy = batch_train_accuracy
        self.batch_train_loss = batch_train_loss
        self.batch_validation_accuracy = batch_validation_accuracy
        self.batch_validation_loss = batch_validation_loss

    def get_batch_num(self):
        return self.batch_num

    def get_batch_train_acc(self):
        return self.batch_train_accuracy

    def get_batch_train_loss(self):
        return self.batch_train_loss

    def get_batch_validation_acc(self):
        return self.batch_validation_accuracy

    def get_batch_validation_loss(self):
        return self.batch_validation_loss

    def set_batch_train_acc(self, acc):
        if 0 <= acc <= 100:
            self.batch_train_accuracy = acc
        else:
            msg = "Batch training accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_batch_train_loss(self, loss):
        self.batch_train_loss = loss

    def set_batch_validation_acc(self, acc):
        if acc is None or 0 <= acc <= 100:  # allow for None in case validation metrics are NOT computed for efficiency
            self.batch_validation_accuracy = acc
        else:
            msg = "Batch validation accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_batch_validation_loss(self, loss):
        self.batch_validation_loss = loss


class EpochStatistics:
    """
    Contains the statistics computed for an Epoch
    """
    def __init__(self, epoch_num):
        self.epoch_num = epoch_num
        self.batch_stats_list = []

    def add_batch(self, batch_stats: Union[BatchStatistics, Sequence[BatchStatistics]]):
        if isinstance(batch_stats, collections.abc.Sequence):
            self.batch_stats_list.extend(batch_stats)
        else:
            self.batch_stats_list.append(batch_stats)

    def get_epoch_num(self):
        return self.epoch_num

    def get_batch_stats(self):
        return self.batch_stats_list


class TrainingRunStatistics:
    """
    Contains the statistics computed for an entire training run
    TODO:
     [ ] - have another function which returns detailed statistics per epoch in an easily serialized manner
    """
    def __init__(self):
        self.epoch_stats_list = []

        self.final_train_acc = 0.
        self.final_train_loss = 0.
        self.final_val_acc = 0.
        self.final_val_loss = 0.
        self.final_clean_data_test_acc = 0.
        self.final_clean_data_n_total = 0
        self.final_triggered_data_test_acc = None
        self.final_triggered_data_n_total = None

    def add_epoch(self, epoch_stats: Union[EpochStatistics, Sequence[EpochStatistics]]):
        if isinstance(epoch_stats, collections.abc.Sequence):
            self.epoch_stats_list.extend(epoch_stats)
        else:
            self.epoch_stats_list.append(epoch_stats)

    def get_epochs_stats(self):
        return self.epoch_stats_list

    def autopopulate_final_summary_stats(self):
        """
        Uses the information from the final epoch's final batch to auto-populate the following statistics:
            final_train_acc
            final_train_loss
            final_val_acc
            final_val_loss
        """
        final_epoch_stats = self.epoch_stats_list[-1]
        final_batch_stats = final_epoch_stats.get_batch_stats()[-1]
        self.set_final_train_acc(final_batch_stats.get_batch_train_acc())
        self.set_final_train_loss(final_batch_stats.get_batch_train_loss())
        self.set_final_val_acc(final_batch_stats.get_batch_validation_acc())
        self.set_final_val_loss(final_batch_stats.get_batch_validation_loss())

    def set_final_train_acc(self, acc):
        if 0 <= acc <= 100:
            self.final_train_acc = acc
        else:
            msg = "Final Training accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_final_train_loss(self, loss):
        self.final_train_loss = loss

    def set_final_val_acc(self, acc):
        if acc is None or 0 <= acc <= 100:  # allow for None in case validation metrics are not computed
            self.final_val_acc = acc
        else:
            msg = "Final validation accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_final_val_loss(self, loss):
        self.final_val_loss = loss

    def set_final_clean_data_test_acc(self, acc):
        if 0 <= acc <= 100:
            self.final_clean_data_test_acc = acc
        else:
            msg = "Final clean data test accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_final_triggered_data_test_acc(self, acc):
        # we allow None in order to indicate that triggered data wasn't present in this dataset
        if acc is None or 0 <= acc <= 100:
            self.final_triggered_data_test_acc = acc
        else:
            msg = "Final triggered data test accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_final_clean_data_n_total(self, n):
        self.final_clean_data_n_total = n

    def set_final_triggered_data_n_total(self, n):
        if n is None or n > 0:
            self.final_triggered_data_n_total = n
        else:
            msg = "Triggered dataset size must be > 0!"
            logger.error(msg)
            raise ValueError(msg)

    def get_summary(self):
        """
        Returns a dictionary of the summary statistics from the training run
        """
        summary_dict = dict()
        summary_dict['final_train_acc'] = self.final_train_acc
        summary_dict['final_train_loss'] = self.final_train_loss
        summary_dict['final_val_acc'] = self.final_val_acc
        summary_dict['final_val_loss'] = self.final_val_loss
        summary_dict['final_clean_data_test_acc'] = self.final_clean_data_test_acc
        summary_dict['final_triggered_data_test_acc'] = self.final_triggered_data_test_acc
        summary_dict['final_clean_data_n_total'] = self.final_clean_data_n_total
        summary_dict['final_triggered_data_n_total'] = self.final_triggered_data_n_total

        return summary_dict

    def save_summary_to_json(self, json_fname: str) -> None:
        """
        Saves the training summary to a JSON file
        """
        summary_dict = self.get_summary()
        # write it to json
        with open(json_fname, 'w') as fp:
            json.dump(summary_dict, fp)
        logger.info("Wrote summary statistics: %s to %s" % (str(summary_dict), json_fname))

    def save_detailed_stats_to_disk(self, fname: str) -> None:
        """
        Saves all batch statistics for every epoch as a CSV file
        :param fname: filename to save the detailed information to
        :return: None
        """
        output_data = []
        for e in self.epoch_stats_list:
            batches_stats = e.get_batch_stats()
            for batch_num, batch_stats in enumerate(batches_stats):
                row = dict(epoch_number=e.epoch_num,
                           batch_num=batch_num,
                           train_accuracy=batch_stats.get_batch_train_acc(),
                           train_loss=batch_stats.get_batch_train_loss(),
                           val_acc=batch_stats.get_batch_validation_acc(),
                           val_loss=batch_stats.get_batch_validation_loss())
                output_data.append(row)
        # write it as a csv
        if len(output_data) > 0:
            keys = output_data[0].keys()
            with open(fname, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(output_data)
            logger.info("Wrote detailed statistics to %s" % (fname, ))
