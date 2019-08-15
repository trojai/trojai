import json
import os
import re
import logging
import types

import torch
import torch.nn as nn

from .config import RunnerConfig, DefaultOptimizerConfig
from .training_statistics import TrainingRunStatistics
from .default_optimizer import DefaultOptimizer
from .optimizer_interface import OptimizerInterface

logger = logging.getLogger(__name__)


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
        train_data, clean_test_data, triggered_test_data = self.cfg.data.load_data()
        arch_factory_kwargs = {} if self.cfg.arch_factory_kwargs is None else self.cfg.arch_factory_kwargs

        if self.cfg.arch_factory_kwargs_generator is not None:
            arch_factory_kwargs.update(self.cfg.arch_factory_kwargs_generator(locals()))

        model = self.cfg.arch_factory.new_architecture(**arch_factory_kwargs)
        if self.cfg.parallel:
            num_available_gpus = torch.cuda.device_count()
            logger.info("Attempting to use " + str(num_available_gpus) + " GPUs for training!")
            model = nn.DataParallel(model)

        model_stats = TrainingRunStatistics()
        # TODO: this is hacked to deal w/ text data, we need to make this better
        training_cfg_list = []
        if isinstance(train_data, types.GeneratorType):
            for data, optimizer in zip(train_data, self.cfg.optimizer_generator):  # both are generators
                model, epoch_training_stats = optimizer.train(model, data, self.cfg.train_val_split)
                model_stats.add_epoch(epoch_training_stats)
                # add training configuration information to data to be saved
                training_cfg_list.append(self._get_training_cfg(optimizer))
        else:
            optimizer = next(self.cfg.optimizer_generator)
            model, epoch_training_stats = optimizer.train(model, train_data, self.cfg.train_val_split)
            model_stats.add_epoch(epoch_training_stats)
            # add training configuration information to data to be saved
            training_cfg_list.append(self._get_training_cfg(optimizer))

        # NOTE: The test function used here is one corresponding to the last optimizer used for training. An exception
        #  will be raised if no training occurred, but validation code prior to this line should prevent this from
        #  ever happening.
        test_acc = optimizer.test(model, clean_test_data, triggered_test_data)

        # Save model train/test statistics and other relevant information
        model_stats.autopopulate_final_summary_stats()
        model_stats.set_final_clean_data_test_acc(test_acc['clean_accuracy'])
        model_stats.set_final_clean_data_n_total(test_acc['clean_n_total'])
        model_stats.set_final_triggered_data_test_acc(test_acc.get('triggered_accuracy', None))
        model_stats.set_final_triggered_data_n_total(test_acc.get('triggered_n_total', None))
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
        filename = self._increment_filename_if_needed(model_path, filename, extn)
        model.eval()
        model_output_fname = os.path.join(model_path, filename)
        stats_output_fname = os.path.join(stats_path, filename+'.stats.json')

        # NOTE: there are some documented issues with saving PyTorch models that were constructed
        #   with nn.DataParallel, although these seem to be resolved.  This note serves as a guide to
        #   people that if you are experiencing issues with saving models, this block of code may be
        #   the culprit.
        logger.info("Saving trained model to " + str(model_output_fname) + " in PyTorch format.")
        if self.cfg.parallel:
            model = model.module
        torch.save(model, model_output_fname)
        model_training_stats_dict = stats.get_summary()
        for i, cfg in enumerate(training_cfg_list):
            model_training_stats_dict.update({"optimizer_"+str(i): cfg})
        # add experiment configuration to the dictionary which gets printed
        model_training_stats_dict.update(self.persist_info)

        # send the statistics to the logger
        logger.info(str(model_training_stats_dict))

        # save the entire dict as a json object
        with open(stats_output_fname, 'w') as fp:
            json.dump(model_training_stats_dict, fp)

    @staticmethod
    def _increment_filename_if_needed(path: str, filename: str, extn: str) -> str:
        """
        Checks if filename already exists at path, and either adds '_1' to end of filename or increments the number
            on the end if one already is there and is separated by a non-alphanumeric value. e.g. if the file names
            'model.pt' and 'other_model_1.pt' are taken, but specified as the file names used to save the model, they
            will be saved as 'model_1.pt' and 'other_model_2.pt'.
        :param path: (str) path to directory where model is to be saved
        :param filename: (str) intended filename
        :param extn: (str) extension at which file is to be saved
        :return: (str) filename, or updated filename
        """
        if os.path.isfile(os.path.join(path, filename)):
            msg = os.path.join(path, filename) + " already exists, appending numerical id to end of file to preserve " \
                                                 "filename uniqueness!"
            logger.warning(msg)
            filename = filename[:-len(extn)]
            ending = re.search(r'[\W_]\d+$', filename)
            if ending is not None and not filename.isdigit() and ending.group()[1:] != '0':
                filename = filename[:ending.start() + 1] + str(int(ending.group()[1:]) + 1) + extn
            else:
                filename += '_1' + extn
        return filename
