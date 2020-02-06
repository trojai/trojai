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
    Represents the statistics collected from training a batch
    NOTE: this is currently unused!
    """
    def __init__(self, batch_num: int,
                 batch_train_accuracy: float,
                 batch_train_loss: float):
        """
        :param batch_num: (int) batch number of collected statistics
        :param batch_train_accuracy: (float) training set accuracy for this batch
        :param batch_train_loss: (float) training loss for this batch
        """
        self.batch_num = batch_num
        self.batch_train_accuracy = batch_train_accuracy
        self.batch_train_loss = batch_train_loss

    def get_batch_num(self):
        return self.batch_num

    def get_batch_train_acc(self):
        return self.batch_train_accuracy

    def get_batch_train_loss(self):
        return self.batch_train_loss

    def set_batch_train_acc(self, acc):
        if 0 <= acc <= 100:
            self.batch_train_accuracy = acc
        else:
            msg = "Batch training accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_batch_train_loss(self, loss):
        self.batch_train_loss = loss


class EpochTrainStatistics:
    """
    Defines the training statistics for one epoch of training
    """
    def __init__(self, train_acc: float, train_loss: float):
        self.train_acc = train_acc
        self.train_loss = train_loss

        self.validate()

    def validate(self):
        if not isinstance(self.train_acc, float):
            msg = "train_acc must be a float, got type {}".format(type(self.train_acc))
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.train_loss, float):
            msg = "train_loss must be a float, got type {}".format(type(self.train_loss))
            logger.error(msg)
            raise ValueError(msg)

    def get_train_acc(self):
        return self.train_acc

    def get_train_loss(self):
        return self.train_loss


class EpochValidationStatistics:
    """
    Defines the validation statistics for one epoch of training
    """
    def __init__(self, val_acc, val_loss):
        self.val_acc = val_acc
        self.val_loss = val_loss

        self.validate()

    def validate(self):
        if not isinstance(self.val_acc, float):
            msg = "val_acc must be a float, got type {}".format(type(self.val_acc))
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.val_loss, float):
            msg = "val_loss must be a float, got type {}".format(type(self.val_loss))
            logger.error(msg)
            raise ValueError(msg)

    def get_val_acc(self):
        return self.val_acc

    def get_val_loss(self):
        return self.val_loss

    def __repr__(self):
        return '(%0.04f, %0.04f)' % (self.val_acc, self.val_loss)


class EpochStatistics:
    """
    Contains the statistics computed for an Epoch
    """
    def __init__(self, epoch_num, training_stats=None, validation_stats=None, batch_training_stats=None):
        self.epoch_num = epoch_num
        if not batch_training_stats:
            self.batch_training_stats = []
        self.epoch_training_stats = training_stats
        self.epoch_validation_stats = validation_stats

        self.validate()

    def add_batch(self, batches: Union[BatchStatistics, Sequence[BatchStatistics]]):
        if isinstance(batches, collections.abc.Sequence):
            self.batch_training_stats.extend(batches)
        else:
            self.batch_training_stats.append(batches)

    def get_batch_stats(self):
        return self.batch_training_stats

    def validate(self):
        if not isinstance(self.batch_training_stats, collections.abc.Sequence):
            msg = "batch_training_stats must be None or a list of BatchTrainingStats objects! " \
                  "Got {}".format(self.batch_training_stats)
            logger.error(msg)
            raise ValueError(msg)
        if self.epoch_training_stats and not isinstance(self.epoch_training_stats, EpochTrainStatistics):
            msg = "training_stats must be None or of type: EpochTrainStatistics!, got type " \
                  "{}".format(type(self.epoch_training_stats))
            logger.error(msg)
            raise ValueError(msg)
        if self.epoch_validation_stats and not isinstance(self.epoch_validation_stats, EpochValidationStatistics):
            msg = "validation_stats must be None or of type: EpochValidationStatistics! Instead got type " \
                  "{}".format(type(self.epoch_validation_stats))
            logger.error(msg)
            raise ValueError(msg)

    def get_epoch_num(self):
        return self.epoch_num

    def get_epoch_training_stats(self):
        return self.epoch_training_stats

    def get_epoch_validation_stats(self):
        return self.epoch_validation_stats


class TrainingRunStatistics:
    """
    Contains the statistics computed for an entire training run, a sequence of epochs
    TODO:
     [ ] - have another function which returns detailed statistics per epoch in an easily serialized manner
    """
    def __init__(self):
        self.stats_per_epoch_list = []

        self.num_epochs_trained_per_optimizer = []

        self.final_train_acc = 0.
        self.final_train_loss = 0.
        self.final_val_acc = 0.
        self.final_val_loss = 0.
        self.final_clean_data_test_acc = 0.
        self.final_clean_data_n_total = 0
        self.final_triggered_data_test_acc = None
        self.final_triggered_data_n_total = None
        self.final_clean_data_triggered_labels_test_acc = None
        self.final_clean_data_triggered_labels_n_total = None
        self.final_optimizer_num_epochs_trained = 0

    def add_epoch(self, epoch_stats: Union[EpochStatistics, Sequence[EpochStatistics]]):
        if isinstance(epoch_stats, collections.abc.Sequence):
            self.stats_per_epoch_list.extend(epoch_stats)
        else:
            self.stats_per_epoch_list.append(epoch_stats)

    def add_num_epochs_trained(self, num_epochs):
        self.num_epochs_trained_per_optimizer.append(num_epochs)

    def get_epochs_stats(self):
        return self.stats_per_epoch_list

    def autopopulate_final_summary_stats(self):
        """
        Uses the information from the final epoch's final batch to auto-populate the following statistics:
            final_train_acc
            final_train_loss
            final_val_acc
            final_val_loss
        """
        final_epoch_training_stats = self.stats_per_epoch_list[-1]

        self.set_final_train_acc(final_epoch_training_stats.get_epoch_training_stats().get_train_acc())
        self.set_final_train_loss(final_epoch_training_stats.get_epoch_training_stats().get_train_loss())
        if final_epoch_training_stats.get_epoch_validation_stats():
            self.set_final_val_acc(final_epoch_training_stats.get_epoch_validation_stats().get_val_acc())
            self.set_final_val_loss(final_epoch_training_stats.get_epoch_validation_stats().get_val_loss())
        self.final_optimizer_num_epochs_trained = self.num_epochs_trained_per_optimizer[-1]

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

    def set_final_clean_data_triggered_label_test_acc(self, acc):
        if acc is None or 0 <= acc <= 100:
            self.final_clean_data_triggered_labels_test_acc = acc
        else:
            msg = "Final clean data test accuracy should be between 0 and 100!"
            logger.error(msg)
            raise ValueError(msg)

    def set_final_clean_data_n_total(self, n):
        self.final_clean_data_n_total = n

    def set_final_triggered_data_n_total(self, n):
        if n is None or n > 0:
            self.final_triggered_data_n_total = n
        else:
            msg = "Triggered dataset size must be > 0! - received dataset.size()=" + str(n)
            logger.error(msg)
            raise ValueError(msg)

    def set_final_clean_data_triggered_label_n(self, n):
        if n is None or n > 0:
            self.final_clean_data_triggered_labels_n_total = n
        else:
            msg = "Triggered dataset size must be > 0! - received dataset.size()=" + str(n)
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
        summary_dict['clean_test_triggered_label_accuracy'] = self.final_clean_data_triggered_labels_test_acc
        summary_dict['clean_test_triggered_label_n_total'] = self.final_clean_data_triggered_labels_n_total
        summary_dict['final_optimizer_num_epochs_trained'] = self.num_epochs_trained_per_optimizer

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
        keys = ['epoch_number', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
        with open(fname, 'w') as output_file:
            # write header first
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            for ii, e in enumerate(self.stats_per_epoch_list):
                # TODO: we ignore batch_statistics for now, we may want to add this in in the future
                epoch_training_stats = e.get_epoch_training_stats()
                epoch_val_stats = e.get_epoch_validation_stats()
                val_acc = None
                val_loss = None
                if epoch_val_stats is not None:
                    val_acc = epoch_val_stats.get_val_acc()
                    val_loss = epoch_val_stats.get_val_loss()
                dict_writer.writerow(dict(epoch_number=e.get_epoch_num(),
                                          train_acc=epoch_training_stats.get_train_acc(),
                                          train_loss=epoch_training_stats.get_train_loss(),
                                          val_acc=val_acc,
                                          val_loss=val_loss))

            logger.info("Wrote detailed statistics to %s" % (fname,))
