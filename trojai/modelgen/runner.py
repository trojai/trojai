import json
import os
import logging
import types
import glob
import uuid
import time
import collections.abc

import numpy as np
import torch
import torch.nn as nn

from .config import RunnerConfig, DefaultOptimizerConfig
from .training_statistics import TrainingRunStatistics
from .default_optimizer import DefaultOptimizer
from .optimizer_interface import OptimizerInterface
from .utils import make_trojai_model_dict

logger = logging.getLogger(__name__)


def try_force_json(x):
    """
    Tries to make a value JSON serializable
    """
    try:
        json.dumps(x)
        return x
    except (TypeError, OverflowError):
        # try to see if datatypes can be converted before giving up
        if isinstance(x, torch.Tensor):
            x = x.data.cpu().numpy().tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()
        elif callable(x):
            x = str(x)
        try:
            json.dumps(x)
            return x
        except (TypeError, OverflowError):
            return None


def try_serialize(d, u):
    # adapted from: https://stackoverflow.com/a/3233356/1057098
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = try_serialize(d.get(k, {}), v)
        else:
            v_new = try_force_json(v)
            if v_new is not None:
                d[k] = v_new
    return d


def add_numerical_extension(path, filename):
    # check if any files already exist in that directory w/ digit extensions or not, and get the filename of interest
    existing_fnames = glob.glob(os.path.join(path, filename + '.*'))
    if len(existing_fnames) > 0:
        # remove the .json & csv files from consideration
        existing_fnames = [os.path.basename(x) for x in existing_fnames if '.json' not in x]
        existing_fnames = [os.path.basename(x) for x in existing_fnames if '.csv' not in x]
        max_cur_digit_ext = 1
        max_cur_digit_fname_without_ext, _ = os.path.splitext(existing_fnames[0])
        # iterate through the filenames and find the maximum integer extension
        for f in existing_fnames:
            fname_without_ext, ext = os.path.splitext(f)
            try:
                ext_val = int(ext[1:])  # the [1:] is needed to remove the . from the extension
                if ext_val > max_cur_digit_ext:
                    max_cur_digit_ext = ext_val
                    max_cur_digit_fname_without_ext = fname_without_ext
            except ValueError:
                pass
        next_digit_ext = max_cur_digit_ext + 1
        fname_to_return = max_cur_digit_fname_without_ext + '.' + (str(next_digit_ext))
    else:
        fname_without_ext, ext = os.path.splitext(filename)
        try:
            cur_digit_ext = int(ext[1:])  # the [1:] is needed to remove the . from the extension
            next_digit_ext = cur_digit_ext + 1
            fname_to_return = fname_without_ext + '.' + (str(next_digit_ext))
        except ValueError:
            fname_to_return = filename + '.1'

    return fname_to_return


class Runner:
    """
    Fundamental unit of model generation, which trains a model as specified in a RunnerConfig object.
    """

    def __init__(self, runner_cfg: RunnerConfig,
                 persist_metadata: dict = None):
        """
        Initialize a model runner, which sets up the Optimizer, passes data to the optimizer, and collects the
        trained model and associated statistics
        :param runner_cfg: (RunnerConfig) Object that contains necessary data and objects to train a model using
            this runner.
        :param persist_metadata: (dict), if not None, the contents of this are appended to the output summary
        dictionary. This can allow for easy tracking of results if they are being collated by an additional process.
        """
        if not isinstance(runner_cfg, RunnerConfig):
            msg = "Expected a RunnerConfig object for argument 'runner_config', instead got " \
                  "type {}".format(type(runner_cfg))
            logger.error(msg)
            raise TypeError(msg)
        self.cfg = runner_cfg
        # todo: make this a type check like with runner_cfg? To reduce confusion if metadata is not a dict but code
        #   runs; make warning
        if persist_metadata is None or not isinstance(persist_metadata, dict):
            msg = "Argument 'persist_metadata' was not None nor type 'dict'. Argument will be ignored."
            logger.warning(msg)
            self.persist_info = {}
        else:
            self.persist_info = persist_metadata

    def run(self) -> None:
        """Trains a model and saves it and the associated model statistics"""
        train_data, clean_test_data, triggered_test_data, clean_test_triggered_labels_data, \
        train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc, clean_test_triggered_labels_desc \
            = self.cfg.data.load_data()
        arch_factory_kwargs = {} if self.cfg.arch_factory_kwargs is None else self.cfg.arch_factory_kwargs
        train_dataloader_kwargs = self.cfg.data.train_dataloader_kwargs
        test_dataloader_kwargs = self.cfg.data.test_dataloader_kwargs

        if self.cfg.arch_factory_kwargs_generator is not None:
            arch_factory_kwargs.update(self.cfg.arch_factory_kwargs_generator(train_dataset_desc,
                                                                              clean_test_dataset_desc,
                                                                              triggered_test_dataset_desc))

        model = self.cfg.arch_factory.new_architecture(**arch_factory_kwargs)
        if self.cfg.parallel:
            num_available_gpus = torch.cuda.device_count()
            logger.info("Attempting to use " + str(num_available_gpus) + " GPUs for training!")
            model = nn.DataParallel(model)

        model_stats = TrainingRunStatistics()
        # TODO: this is hacked to deal w/ text data, we need to make this better
        training_cfg_list = []
        t1 = time.time()
        if isinstance(train_data, types.GeneratorType):
            for data, optimizer in zip(train_data, self.cfg.optimizer_generator):  # both are generators
                model, epoch_training_stats, num_epochs_trained, best_val_epoch = \
                    optimizer.train(model, data, train_dataloader_kwargs, use_amp=self.cfg.amp)
                model_stats.add_epoch(epoch_training_stats)
                model_stats.add_num_epochs_trained(num_epochs_trained)
                model_stats.add_best_epoch_val(best_val_epoch)
                # add training configuration information to data to be saved
                training_cfg_list.append(self._get_training_cfg(optimizer))
        else:
            optimizer = next(self.cfg.optimizer_generator)
            model, training_stats, num_epochs_trained, best_val_epoch = \
                optimizer.train(model, train_data, train_dataloader_kwargs, use_amp=self.cfg.amp)
            model_stats.add_epoch(training_stats)
            model_stats.add_num_epochs_trained(num_epochs_trained)
            model_stats.add_best_epoch_val(best_val_epoch)
            # add training configuration information to data to be saved
            training_cfg_list.append(self._get_training_cfg(optimizer))
        t2 = time.time()
        # NOTE: The test function used here is one corresponding to the last optimizer used for training. An exception
        #  will be raised if no training occurred, but validation code prior to this line should prevent this from
        #  ever happening.
        test_acc = optimizer.test(model, clean_test_data, triggered_test_data, clean_test_triggered_labels_data, test_dataloader_kwargs)
        t3 = time.time()

        # Save model train/test statistics and other relevant information
        model_stats.autopopulate_final_summary_stats()
        model_stats.set_final_clean_data_test_acc(test_acc['clean_accuracy'])
        model_stats.set_final_clean_data_n_total(test_acc['clean_n_total'])
        model_stats.set_final_triggered_data_test_acc(test_acc.get('triggered_accuracy', None))
        model_stats.set_final_triggered_data_n_total(test_acc.get('triggered_n_total', None))
        model_stats.set_final_clean_data_triggered_label_test_acc(
            test_acc.get('clean_test_triggered_label_accuracy', None))
        model_stats.set_final_clean_data_triggered_label_n(test_acc.get('clean_test_triggered_label_n_total', None))

        # add training/test wall-times to stats
        self.persist_info['training_wall_time_sec'] = t2 - t1
        self.persist_info['test_wall_time_sec'] = t3 - t2

        self._save_model_and_stats(model, model_stats, training_cfg_list)

    @staticmethod
    def _get_training_cfg(optimizer):
        if isinstance(optimizer, DefaultOptimizerConfig):
            training_cfg = optimizer.training_cfg.get_cfg_as_dict()
        elif isinstance(optimizer, DefaultOptimizer) or isinstance(optimizer, OptimizerInterface):
            training_cfg = optimizer.get_cfg_as_dict()
        else:
            msg = "Unable to get training_cfg from optimizer(_cfg): {}, returning empty dict".format(optimizer)
            logger.warning(msg)
            training_cfg = dict()
        return training_cfg

    def _save_model_and_stats(self, model: nn.Module, stats: TrainingRunStatistics, training_cfg_list: list):
        model_path = self.cfg.model_save_dir
        if not os.path.isdir(model_path):
            try:
                os.makedirs(model_path)
            except IOError as e:
                logger.error(e)
        stats_path = self.cfg.stats_save_dir
        if not os.path.isdir(model_path):
            try:
                os.makedirs(model_path)
            except IOError as e:
                logger.error(e)

        extn = '.pt'

        if self.cfg.filename is not None:
            filename = self.cfg.filename
            if os.path.splitext(filename)[1] != extn:
                filename += extn
        else:
            if self.cfg.run_id is not None:
                filename = model.__class__.__name__ + '_id' + str(self.cfg.run_id)
            else:
                filename = model.__class__.__name__
            if self.persist_info is not None and 'name' in self.persist_info:
                filename += '_' + self.persist_info['name']
            filename += extn

        if self.cfg.save_with_hash:
            filename += '.' + str(uuid.uuid1().hex)
        else:
            filename = add_numerical_extension(model_path, filename)

        model.eval()
        model_output_fname = os.path.join(model_path, filename)
        stats_output_fname = os.path.join(stats_path, filename + '.stats.json')
        detailed_stats_output_fname = os.path.join(stats_path, filename + '.stats.detailed.csv')

        logger.info("Saving trained model to " + str(model_output_fname) + " in PyTorch format.")
        if self.cfg.parallel:
            model = model.module
        model.cpu()  # move to cpu before saving to simplify loading the model
        if self.cfg.model_save_format == 'pt':
            torch.save(model, model_output_fname)
        elif self.cfg.model_save_format == 'state_dict':
            save_dict = make_trojai_model_dict(model)
            torch.save(save_dict, model_output_fname)
        model_training_stats_dict = stats.get_summary()
        for i, cfg in enumerate(training_cfg_list):
            # remove function handles from the training_cfg which have been copied over
            cfg.pop('val_data_transform', None)
            cfg.pop('val_label_transform', None)
            model_training_stats_dict.update({"optimizer_" + str(i): cfg})
        # add experiment configuration to the dictionary which gets printed
        model_training_stats_dict.update(self.persist_info)

        # try to make every value JSON Serializable
        model_training_stats_serialized = dict()
        model_training_stats_serialized = try_serialize(model_training_stats_serialized, model_training_stats_dict)

        # send the statistics to the logger
        logger.info(str(model_training_stats_serialized))

        # save the entire dict as a json object
        with open(stats_output_fname, 'w') as fp:
            json.dump(model_training_stats_serialized, fp, indent=2)
        # save detailed statistics
        stats.save_detailed_stats_to_disk(detailed_stats_output_fname)
