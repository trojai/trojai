import collections.abc
import copy
import importlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, Sequence, Any
import math

import cloudpickle as pickle
import numpy as np
import torch

from .architecture_factory import ArchitectureFactory
from .constants import VALID_LOSS_FUNCTIONS, VALID_DEVICES, VALID_OPTIMIZERS
from .data_manager import DataManager
from .optimizer_interface import OptimizerInterface

logger = logging.getLogger(__name__)

"""
Defines all configurations pertinent to model generation.
"""


class ConfigInterface(ABC):
    """
    Defines the interface for all configuration objects
    """

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        pass


class OptimizerConfigInterface(ConfigInterface):
    @abstractmethod
    def get_device_type(self):
        pass

    def save(self, fname):
        pass

    @staticmethod
    @abstractmethod
    def load(fname):
        pass


class EarlyStoppingConfig(ConfigInterface):
    """
    Defines configuration related to early stopping.
    """
    def __init__(self, num_epochs: int = 5, val_loss_eps: float = 1e-3):
        """
        :param num_epochs: the # of epochs for which to monitor the validation accuracy over
        :param val_loss_eps: the threshold between the validation loss for the # of epochs to monitor the
                before deciding to perform early stopping
        """
        self.num_epochs = num_epochs
        self.val_loss_eps = val_loss_eps

        self.validate()

    def validate(self):
        if not isinstance(self.num_epochs, int) or self.num_epochs < 2:
            msg = "num_epochs to monitor must be an integer > 1!"
            logger.error(msg)
            raise ValueError(msg)
        try:
            self.val_loss_eps = float(self.val_loss_eps)
        except ValueError:
            msg = "val_loss_eps must be a float"
            logger.error(msg)
            raise ValueError(msg)
        if self.val_loss_eps < 0:
            msg = "val_loss_eps must be >= 0!"
            logger.error(msg)
            raise ValueError(msg)

    def __deepcopy__(self, memodict={}):
        return EarlyStoppingConfig(self.num_epochs, self.val_loss_eps)

    def __eq__(self, other):
        if self.num_epochs == other.num_epochs and math.isclose(self.val_loss_eps, other.val_acc_eps):
            return True
        else:
            return False

    def __str__(self):
        return "ES[%d:%0.02f]" % (self.num_epochs, self.val_loss_eps)


class TrainingConfig(ConfigInterface):
    """
    Defines all required items to setup training with an optimizer
    """

    def __init__(self,
                 device: Union[str, torch.device] = 'cpu',
                 epochs: int = 10,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 optim: Union[str, OptimizerInterface] = 'adam',
                 optim_kwargs: dict = None,
                 objective: Union[str, Callable] = 'cross_entropy_loss',
                 save_best_model: bool = False,
                 train_val_split: float = 0.,
                 val_data_transform: Callable[[Any], Any] = lambda x: x,
                 val_label_transform: Callable[[int], int] = lambda y: y,
                 val_dataloader_kwargs: dict = None,
                 early_stopping: EarlyStoppingConfig = None) -> None:
        """
        Initializes a TrainingConfig object
        :param device: string or torch.device object representing the device on which computation will be performed
        :param epochs: the number of epochs to train the model
        :param batch_size: batch size used to train the model
        :param lr: the learning rate
        :param optim: either one of trojai_private.modelgen.constants.VALID_OPTIMIZERS or an optimizer
                object implementing trojai_private.modelgen.optimizer_interface.OptimizerInterface
        :param optim_kwargs: any additional kwargs to be passed to the optimizer
        :param objective: either one of trojai_private.modelgen.constants.VALID_OBJECTIVES or a
                callable function that can compute a metric given y_hat and y_true
        :param save_best_model: if True, returns the best model as computed by validation accuracy (if computed),
                                else, training accuracy (if validation dataset is not desired).  if False,
                                the model returned by the optimizer will just be the model at the final epoch of
                                training
        :param train_val_split: (float) if > 0, then splits the training dataset and uses it as validation.  If 0
            the training dataset is not split and validation is not computed
        :param val_data_transform: (function: any -> any) how to transform the validation data (e.g. an image) to fit
            into the desired model and objective function; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param val_label_transform: (function: int->int) how to transform the label to the validation data; optional
            NOTE: Currently - this argument is only used if data_type='image'
        :param val_dataloader_kwargs: (dict) Keyword arguments to pass to the torch DataLoader object during for
            validation data. See https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html for more
            documentation. If None, defaults will be used. Defaults depend on the optimizer used, but are likely
            something like:
                {batch_size: <batch size given in training config>, shuffle: False, pin_memory=<decided by optimizer>,
                 drop_last=True}
            NOTE: Setting values in this dictionary that are normally set by the optimizer will override them during
                training. Use with caution. We recommend only using the following keys: 'shuffle', 'num_workers',
                'pin_memory', and 'drop_last'.
        :param early_stopping: configuration for early stopping
        TODO:
         [ ] - allow user to configure what the "best" model is
        """
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.objective = objective
        self.save_best_model = save_best_model
        self.train_val_split = train_val_split
        self.early_stopping = early_stopping
        self.val_data_transform = val_data_transform
        self.val_label_transform = val_label_transform
        self.val_dataloader_kwargs = val_dataloader_kwargs

        if self.optim_kwargs is None:
            self.optim_kwargs = {}

        self.validate()

        # convert to a torch.device object
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def validate(self) -> None:
        """
        Validate the object configuration
        :return: None
        """
        if not isinstance(self.device, torch.device) and self.device not in VALID_DEVICES:
            msg = "device must be either a torch.device object, or one of the following:" + str(VALID_DEVICES)
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.epochs, int) or self.epochs < 1:
            msg = "epochs must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            msg = "batch_size must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.lr, float):
            msg = "lr must be a float!"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.optim, OptimizerInterface) and self.optim not in VALID_OPTIMIZERS:
            msg = "optim must be either a OptimizerInterface object, or one of the following:" + str(VALID_OPTIMIZERS)
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.optim_kwargs, dict):
            msg = "optim_kwargs must be a dictionary!"
            logger.error(msg)
            raise ValueError(msg)
        if not callable(self.objective) and self.objective not in VALID_LOSS_FUNCTIONS:
            msg = "objective must be a callable, or one of the following:" + str(VALID_LOSS_FUNCTIONS)
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.save_best_model, bool):
            msg = "save_best_model must be a boolean!"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.train_val_split, float):
            msg = "train_val_split must a float between 0 and 1!"
            logger.error(msg)
            raise ValueError(msg)
        else:
            if self.train_val_split < 0 or self.train_val_split > 1:
                msg = "train_val_split must be between 0 and 1, inclusive"
                logger.error(msg)
                raise ValueError(msg)
        if self.early_stopping and not isinstance(self.early_stopping, EarlyStoppingConfig):
            msg = "early_stopping must be of type EarlyStoppingConfig or None"
            logger.error(msg)
            raise ValueError(msg)

        # disallow early-stopping and save best model to both be turned on - that doesn't make logical sense
        if self.early_stopping and self.save_best_model:
            msg = "early-stopping and save best model cannot both be on at the same time!"
            logger.error(msg)
            raise ValueError(msg)

        if not callable(self.val_data_transform):
            raise TypeError("Expected a function for argument 'val_data_transform', "
                            "instead got type: {}".format(type(self.val_data_transform)))
        if not callable(self.val_label_transform):
            raise TypeError("Expected a function for argument 'val_label_transform', "
                            "instead got type: {}".format(type(self.val_label_transform)))
        if self.val_dataloader_kwargs is not None and not isinstance(self.val_dataloader_kwargs, dict):
            msg = "val_dataloader_kwargs must be a dictionary!"
            logger.error(msg)
            raise ValueError(msg)

    def get_cfg_as_dict(self):
        """
        Returns a dictionary representation of the configuration
        :return: (dict) a dictionary
        """
        output_dict = dict(device=str(self.device.type),
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           learning_rate=self.lr,
                           optim=self.optim,
                           objective=self.objective,
                           save_best_model=self.save_best_model,
                           early_stopping=str(self.early_stopping),
                           val_data_transform=self.val_data_transform,
                           val_label_transform=self.val_label_transform,
                           val_dataloader_kwargs=self.val_dataloader_kwargs)
        return output_dict

    def __str__(self):
        str_repr = "TrainingConfig: device[%s], num_epochs[%d], batch_size[%d], learning_rate[%.5e], optimizer[%s], " \
                   "objective[%s], train_val_split[%0.02f], val_data_transform[%s], " \
                   "val_label_transform[%s], val_dataloader_kwargs[%s], early_stopping[%s]" % \
                   (str(self.device.type), self.epochs, self.batch_size, self.lr,
                    str(self.optim), str(self.objective), self.train_val_split, str(self.val_data_transform),
                    str(self.val_label_transform), str(self.val_dataloader_kwargs), str(self.early_stopping))
        return str_repr

    def __deepcopy__(self, memodict={}):
        new_device = self.device.type  # copy will keep a string version fo device, so that when
        # it gets instantiated, it will generate a device object
        # on the node
        epochs = self.epochs
        batch_size = self.batch_size
        lr = self.lr
        save_best_model = self.save_best_model
        train_val_split = self.train_val_split
        early_stopping = copy.deepcopy(self.early_stopping)
        val_data_transform = copy.deepcopy(self.val_data_transform)
        val_label_transform = copy.deepcopy(self.val_label_transform)
        val_dataloader_kwargs = copy.deepcopy(self.val_dataloader_kwargs)
        if isinstance(self.optim, str):
            optim = self.optim
        elif isinstance(self.optim, OptimizerInterface):
            optim = copy.deepcopy(self.optim)
        else:
            msg = "The TrainingConfig object you are trying to copy is corrupted!"
            logger.error(msg)
            raise ValueError(msg)
        optim_kwargs = self.optim_kwargs
        if isinstance(self.objective, str):
            objective = self.objective
        elif callable(self.objective):
            objective = copy.deepcopy(self.objective)
        else:
            msg = "The TrainingConfig object you are trying to copy is corrupted!"
            logger.error(msg)
            raise ValueError(msg)
        return TrainingConfig(new_device, epochs, batch_size, lr, optim, optim_kwargs, objective, save_best_model,
                              train_val_split, val_data_transform, val_label_transform, val_dataloader_kwargs,
                              early_stopping)

    def __eq__(self, other):
        if self.device.type == other.device.type and self.epochs == other.epochs and \
           self.batch_size == other.batch_size and self.lr == other.lr and \
           self.save_best_model == other.save_best_model and \
           self.train_val_split == other.train_val_split and \
           self.early_stopping == other.early_stopping and \
           self.val_data_transform == other.val_data_transform and \
           self.val_label_transform == other.val_label_transform and \
           self.val_dataloader_kwargs == other.val_dataloader_kwargs:
            # now check the objects
            if self.optim == other.optim and self.objective == other.objective:
                return True
            else:
                return False
        else:
            return False


class ReportingConfig(ConfigInterface):
    """
    Defines all options to setup how data is reported back to the user while models are being trained
    """

    def __init__(self,
                 num_batches_per_logmsg: int = 100,
                 num_epochs_per_metric: int = 1,
                 num_batches_per_metrics: int = 50,
                 tensorboard_output_dir: str = None,
                 experiment_name: str = 'experiment'):
        """
        Initializes a ReportingConfig object.
        :param num_batches_per_logmsg: The # of batches which are computed before a log message is written.
        :param num_epochs_per_metric: The number of epochs before metrics are computed.
        :param num_batches_per_metrics: The number of batches before metrics are computed.
        :param tensorboard_output_dir: the directory to which tensorboard data should be written.
        :param experiment_name: A string identifier to associate with the configuration.
        """
        self.num_batches_per_logmsg = num_batches_per_logmsg
        self.num_epochs_per_metrics = num_epochs_per_metric
        self.num_batches_per_metrics = num_batches_per_metrics
        self.tensorboard_output_dir = tensorboard_output_dir
        self.experiment_name = experiment_name

        self.validate()

    def validate(self):
        if not isinstance(self.num_batches_per_logmsg, int) or self.num_batches_per_logmsg < 0:
            msg = "num_batches_per_logmsg must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.num_epochs_per_metrics, int) or self.num_epochs_per_metrics < 0:
            msg = "num_epochs_per_metrics must be an integer > 0"
            logger.error(msg)
            raise ValueError(msg)
        if self.num_batches_per_metrics is not None and (not isinstance(self.num_batches_per_metrics, int) or
                                                         self.num_batches_per_metrics < 0):
            msg = "num_batches_per_metrics must be an integer > 0 or None!"
            logger.error(msg)
            raise ValueError(msg)

    def __str__(self):
        str_repr = "ReportingConfig: num_batches/log_msg[%d], num_epochs/metric[%d], num_batches/metric[%d], " \
                   "tensorboard_dir[%s] experiment_name=[%s]" % \
                   (self.num_batches_per_logmsg, self.num_epochs_per_metrics, self.num_batches_per_metrics,
                    self.tensorboard_output_dir, self.experiment_name)
        return str_repr

    def __copy__(self):
        return ReportingConfig(self.num_batches_per_logmsg, self.num_epochs_per_metrics,
                               self.num_batches_per_metrics,
                               self.tensorboard_output_dir, self.experiment_name)

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __eq__(self, other):
        if self.num_batches_per_logmsg == other.num_batches_per_logmsg and \
                self.num_epochs_per_metrics == other.num_epochs_per_metrics and \
                self.num_batches_per_metrics == other.num_batches_per_metrics and \
                self.tensorboard_output_dir == other.tensorboard_output_dir and \
                self.experiment_name == other.experiment_name:
            return True
        else:
            return False


class TorchTextOptimizerConfig(OptimizerConfigInterface):
    """
    Defines the configuration needed to setup the TorchTextOptimizer
    """

    def __init__(self, training_cfg: TrainingConfig = None, reporting_cfg: ReportingConfig = None,
                 copy_pretrained_embeddings: bool = False):
        """
        Initializes a TorchTextOptimizer
        :param training_cfg: a TrainingConfig object, if None, a default TrainingConfig object will be constructed
        :param reporting_cfg: a ReportingConfig object, if None, a default ReportingConfig object will be constructed
        :param copy_pretrained_embeddings: if True, will copy over pretrained embeddings into network from the built
            vocabulary
        """
        self.training_cfg = training_cfg
        self.reporting_cfg = reporting_cfg
        self.copy_pretrained_embeddings = copy_pretrained_embeddings

        self.validate()

    def validate(self):
        if self.training_cfg is None:
            logger.info("Using default training configuration to setup Optimizer!")
            self.training_cfg = TrainingConfig()
        elif not isinstance(self.training_cfg, TrainingConfig):
            msg = "training_cfg must be of type TrainingConfig"
            logger.error(msg)
            raise TypeError(msg)

        if self.reporting_cfg is None:
            logger.info("Using default reporting configuration to setup Optimizer!")
            self.reporting_cfg = ReportingConfig()
        elif not isinstance(self.reporting_cfg, ReportingConfig):
            msg = "reporting_cfg must be of type ReportingConfig"
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(self.copy_pretrained_embeddings, bool):
            msg = "copy_pretrained_embeddings must be a boolean datatype!"
            logger.error(msg)
            raise TypeError(msg)

    def __deepcopy__(self, memodict={}):
        training_cfg_copy = copy.deepcopy(self.training_cfg)
        reporting_cfg_copy = copy.deepcopy(self.reporting_cfg)
        return TorchTextOptimizerConfig(training_cfg_copy, reporting_cfg_copy, self.copy_pretrained_embeddings)

    def __eq__(self, other):
        if self.training_cfg == other.training_cfg and self.reporting_cfg == other.reporting_cfg and \
           self.copy_pretrained_embeddings == other.copy_pretrained_embeddings:
            return True
        else:
            return False

    def save(self, fname):
        """
        Saves the optimizer configuration to a file
        :param fname: the filename to save the config to
        :return: None
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        """
        Loads a configuration from disk
        :param fname: the filename where the config is stored
        :return: the loaded configuration
        """
        with open(fname, 'rb') as f:
            loaded_optimzier_cfg = pickle.load(f)
        return loaded_optimzier_cfg

    def get_device_type(self):
        """
        Returns the device associated w/ this optimizer configuration.  Needed to save/load for UGE.
        :return (str): the device type represented as a string
        """
        return str(self.training_cfg.device)


class DefaultOptimizerConfig(OptimizerConfigInterface):
    """
    Defines the configuration needed to setup the DefaultOptimizer
    """

    def __init__(self, training_cfg: TrainingConfig = None, reporting_cfg: ReportingConfig = None):
        """
        Initializes a Default Optimizer
        :param training_cfg: a TrainingConfig object, if None, a default TrainingConfig object will be constructed
        :param reporting_cfg: a ReportingConfig object, if None, a default ReportingConfig object will be constructed
        """
        if training_cfg is None:
            logger.info("Using default training configuration to setup Optimizer!")
            self.training_cfg = TrainingConfig()
        elif not isinstance(training_cfg, TrainingConfig):
            msg = "training_cfg must be of type TrainingConfig"
            logger.error(msg)
            raise TypeError(msg)
        else:
            self.training_cfg = training_cfg

        if reporting_cfg is None:
            logger.info("Using default reporting configuration to setup Optimizer!")
            self.reporting_cfg = ReportingConfig()
        elif not isinstance(reporting_cfg, ReportingConfig):
            msg = "reporting_cfg must be of type ReportingConfig"
            logger.error(msg)
            raise TypeError(msg)
        else:
            self.reporting_cfg = reporting_cfg

    def __deepcopy__(self, memodict={}):
        training_cfg_copy = copy.deepcopy(self.training_cfg)
        reporting_cfg_copy = copy.deepcopy(self.reporting_cfg)
        return DefaultOptimizerConfig(training_cfg_copy, reporting_cfg_copy)

    def __eq__(self, other):
        if self.training_cfg == other.training_cfg and self.reporting_cfg == other.reporting_cfg:
            return True
        else:
            return False

    def get_device_type(self):
        """
        Returns the device associated w/ this optimizer configuration.  Needed to save/load for UGE.
        :return (str): the device type represented as a string
        """
        return str(self.training_cfg.device)

    def save(self, fname):
        """
        Saves the optimizer configuration to a file
        :param fname: the filename to save the config to
        :return: None
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        """
        Loads a configuration from disk
        :param fname: the filename where the config is stored
        :return: the loaded configuration
        """
        with open(fname, 'rb') as f:
            loaded_optimzier_cfg = pickle.load(f)
        return loaded_optimzier_cfg


class ModelGeneratorConfig(ConfigInterface):
    """Object used to configure the model generator"""

    def __init__(self, arch_factory: ArchitectureFactory, data: DataManager,
                 model_save_dir: str, stats_save_dir: str, num_models: int,
                 arch_factory_kwargs: dict = None, arch_factory_kwargs_generator: Callable = None,
                 optimizer: Union[Union[OptimizerInterface, DefaultOptimizerConfig],
                                  Sequence[Union[OptimizerInterface, DefaultOptimizerConfig]]] = None,
                 parallel=False,
                 experiment_cfg: dict = None,
                 run_ids: Union[Any, Sequence[Any]] = None,
                 filenames: Union[str, Sequence[str]] = None,
                 save_with_hash: bool = False):
        """
        Initializes the ModelGeneratorConfig object which provides needed information for generating models for a given
        experiment.

        :param arch_factory: ArchitectureFactory object that provides instantiated
            architectures (untrained models) to be trained on the data.
        :param data: TrojaiDataManager object containing the experiment path and files.
        :param model_save_dir: path to directory where the models should be saved
        :param stats_save_dir: path to directory where the model training stats should be saved
        :param num_models: number of models to train with this configuration
        :param arch_factory_kwargs: (dict) a dictionary which contains keywords and associated values
            that are needed to instantiate a trainable module from the factory
        :param arch_factory_kwargs_generator: (callable) a callable, or None, which takes a dictionary of all
            variables defined in the Runner's namespace, and then creates a new dictionary that contains the keyword
            arguments to instantiate an architecture from the architecture factory
        :param optimizer: a OptimizerInterface object, or a DefaultOptimizer configuration, or possibly mixed sequence
            of both.  If a sequence of optimizers is passed, then the length of that sequence must match the number
            of sequential datasets that are to be used for training the model.
        :param parallel: (bool) - if True, attempts to use multiple GPU's
        :param experiment_cfg: dictionary containing information regarding the experiment which is being run by the
            ModelGenerator.  This information is also saved in the output summary JSON file that is associated with
            every model that is generated.
        :param run_ids: Identifiers for models. If a sequence, len(run_ids) must be equal to num_models
        :param filenames: An optional list of file names to save each model by each
            file name, or a single filename to have models be saved with the same file name with '_#' added to
            the end, e.g. 'filename.pt', 'filename_1.pt', 'filename_2.pt', ...
            If this argument is not provided, then models generated will be saved with filenames indicated by the
            experiment name in the experiment_cfg dictionary
        :param save_with_hash: (bool) if True, appends a hash to the end of a filename to prevent any conflicts from
            occurring w.r.t. filenames.  This can be useful if you are using a cluster environment and the filesystem
            across nodes takes time to replicate
        """
        self.arch_factory = arch_factory
        self.arch_factory_kwargs = arch_factory_kwargs
        self.arch_factory_kwargs_generator = arch_factory_kwargs_generator
        self.data = data
        self.model_save_dir = model_save_dir
        self.stats_save_dir = stats_save_dir
        self.num_models = num_models

        self.optimizer = optimizer
        self.parallel = parallel
        self.experiment_cfg = dict() if experiment_cfg is None else experiment_cfg

        self.run_ids = run_ids  # it might be useful to allow something like a generator for this argument
        self.filenames = filenames  # it might be useful to allow something like a generator for this argument
        self.save_with_hash = save_with_hash

        self.validate()

    def __deepcopy__(self, memodict={}):
        arch_factory_copy = copy.deepcopy(
            self.arch_factory)  # I think this is OK b/c the ArchFactory is a class definition
        data_copy = copy.deepcopy(self.data)  # the default should work properly here b/c all properties are primitives
        optimizer_copy = copy.deepcopy(self.optimizer)
        return ModelGeneratorConfig(arch_factory_copy, data_copy,
                                    self.model_save_dir, self.stats_save_dir, self.num_models,
                                    self.arch_factory_kwargs, self.arch_factory_kwargs_generator,
                                    optimizer_copy, self.parallel, self.experiment_cfg,
                                    self.run_ids, self.filenames, self.save_with_hash)

    def __eq__(self, other):
        if self.arch_factory == other.arch_factory and self.data == other.data and self.optimizer == other.optimizer \
                and self.parallel == other.parallel \
                and self.model_save_dir == other.model_save_dir and self.stats_save_dir == other.stats_save_dir \
                and self.arch_factory_kwargs == other.arch_factory_kwargs \
                and self.arch_factory_kwargs_generator == other.arch_factory_kwargs_generator \
                and self.experiment_cfg == other.experiment_cfg and self.run_ids == other.run_ids \
                and self.filenames == other.filenames and self.save_with_hash == other.save_with_hash:
            return True
        else:
            return False

    def validate(self) -> None:
        """
        Validate the input arguments to construct the object
        :return: None
        """
        if not (isinstance(self.arch_factory, ArchitectureFactory)):
            msg = "Expected an ArchitectureFactory object for argument 'architecture_factory', " \
                  "instead got type: {}".format(type(self.arch_factory))
            logger.error(msg)
            raise TypeError(msg)
        if self.arch_factory_kwargs is not None and not isinstance(self.arch_factory_kwargs, dict):
            msg = "Expected dictionary for arch_factory_kwargs"
            logger.error(msg)
            raise TypeError(msg)
        if self.arch_factory_kwargs_generator is not None and not callable(self.arch_factory_kwargs_generator):
            msg = "arch_factory_kwargs_generator must be a Callable!"
            logger.error(msg)
            raise TypeError(msg)
        if not (isinstance(self.data, DataManager)):
            msg = "Expected an TrojaiDataManager object for argument 'data', " \
                  "instead got type: {}".format(type(self.data))
            logger.error(msg)
            raise TypeError(msg)
        if not type(self.model_save_dir) == str:
            msg = "Expected type 'string' for argument 'model_save_dir, instead got type: " \
                  "{}".format(type(self.model_save_dir))
            logger.error(msg)
            raise TypeError(msg)
        if not os.path.isdir(self.model_save_dir):
            try:
                os.makedirs(self.model_save_dir)
            except IOError as e:
                msg = "'model_save_dir' was not found and could not be created" \
                      "...\n{}".format(e.__traceback__)
                logger.error(msg)
                raise IOError(msg)
        if not type(self.num_models) == int:
            msg = "Expected type 'int' for argument 'num_models, instead got type: " \
                  "{}".format(type(self.num_models))
            logger.error(msg)
            raise TypeError(msg)
        if self.filenames is not None:
            if isinstance(self.filenames, Sequence):
                for filename in self.filenames:
                    if not type(filename) == str:
                        msg = "Encountered non-string in argument 'filenames': {}".format(filename)
                        logger.error(msg)
                        raise TypeError(msg)
            else:
                if not isinstance(self.filenames, str):
                    msg = "Filename provided as prefix must be of type string!"
                    logger.error(msg)
                    raise TypeError(msg)
        if self.run_ids is not None and len(self.run_ids) != self.num_models:
            msg = "Argument 'run_ids' was provided, but len(run_ids) != num_models"
            logger.error(msg)
            raise RuntimeError(msg)
        if self.filenames is not None and len(self.filenames) != self.num_models:
            msg = "Argument 'filenames' was provided, but len(filenames) != num_models"
            logger.error(msg)
            raise RuntimeError(msg)
        if self.run_ids is not None and self.filenames is not None:
            msg = "Argument 'filenames' was provided with argument 'run_ids', 'run_ids' will be ignored..."
            logger.warning(msg)
        if not isinstance(self.save_with_hash, bool):
            msg = "Expected boolean for save_with_hash argument"
            logger.error(msg)
            raise ValueError(msg)

        RunnerConfig.validate_optimizer(self.optimizer, self.data)

        if not isinstance(self.parallel, bool):
            msg = "parallel argument must be a boolean!"
            logger.error(msg)
            raise ValueError(msg)

    def __getstate__(self):
        """
        Function which dictates which objects will be saved when pickling the ModelGeneratorConfig object.  This is
        only useful for the UGEModelGenerator, which needs to save the data before parallelizing a job.
        :return: a dictionary of the state of the ModelGeneratorConfig object.
        """
        return {'arch_factory': self.arch_factory,
                'data': self.data,
                'model_save_dir': self.model_save_dir,
                'stats_save_dir': self.stats_save_dir,
                'num_models': self.num_models,
                'arch_factory_kwargs': self.arch_factory_kwargs,
                'arch_factory_kwargs_generator': self.arch_factory_kwargs_generator,
                'parallel': self.parallel,
                'experiment_cfg': self.experiment_cfg,
                'run_ids': self.run_ids,
                'filenames': self.filenames,
                'save_with_hash': self.save_with_hash
                }

    def save(self, fname: str):
        """
        Saves the ModelGeneratorConfig object in two different parts.  Every object within the config, except for the
        optimizer is saved in the .klass.save file, and the optimizer is saved separately.
        :param fname - the filename to save the configuration to
        :return: None
        """
        # we save optimizer and the remainder of the components separately
        optimizer_klass_save_fname = fname + '.optimizer.klass.save'
        optimizer_save_fname = fname + '.optimizer.save'
        remainder_data_save_fname = fname + '.arch_data.save'

        with open(remainder_data_save_fname, 'wb') as f:
            pickle.dump(self, f)
        # save the optimizer class name, so we can load it properly
        optimizer_klass_name = '.'.join([self.optimizer.__module__, self.optimizer.__class__.__name__])
        with open(optimizer_klass_save_fname, 'w') as f:
            f.write(optimizer_klass_name)
        self.optimizer.save(optimizer_save_fname)

    @staticmethod
    def load(fname: str):
        """
        Loads a saved modelgen_cfg object from data that was saved using the .save() function.
        :param fname: the filename where the modelgen_cfg object is saved
        :return: a ModelGeneratorConfig object
        """
        optimizer_klass_save_fname = fname + '.optimizer.klass.save'
        optimizer_save_fname = fname + '.optimizer.save'
        remainder_data_save_fname = fname + '.arch_data.save'

        with open(remainder_data_save_fname, 'rb') as f:
            modelgen_cfg = pickle.load(f)

        # load the class name of the optimizer that was used
        with open(optimizer_klass_save_fname, 'r') as f:
            optimizer_module_and_klass_name = f.readline()
        # load the module
        ss = optimizer_module_and_klass_name.split('.')
        optimizer_module_name = '.'.join(ss[0:-1])
        optimizer_klass_name = ss[-1]
        optimizer_module = importlib.import_module(optimizer_module_name)
        optimizer_klass_def = getattr(optimizer_module, optimizer_klass_name)
        optimizer_load = optimizer_klass_def.load(optimizer_save_fname)

        # reconstruct the ModelGeneratorConfig object
        modelgen_cfg.optimizer = optimizer_load
        modelgen_cfg.validate()
        return modelgen_cfg


class RunnerConfig(ConfigInterface):
    """
    Container for all parameters needed to use the Runner to train a model.
    """

    def __init__(self, arch_factory: ArchitectureFactory, data: DataManager,
                 arch_factory_kwargs: dict = None, arch_factory_kwargs_generator: Callable = None,
                 optimizer: Union[OptimizerInterface, DefaultOptimizerConfig,
                                  Sequence[Union[OptimizerInterface, DefaultOptimizerConfig]]] = None,
                 parallel: bool = False,
                 model_save_dir: str = "/tmp/models", stats_save_dir: str = "/tmp/model_stats",
                 model_save_format: str = "pt",
                 run_id: Any = None, filename: str = None, save_with_hash: bool = False):
        """
        Initialize a RunnerConfig object
        :param arch_factory: (Architecture Factory) a trainable Pytorch module generator.
        :param data: (TrojaiDataManager) a TrojaiDataManager object containing the paths to the data being trained and
            tested on, as well as functions dictating how the data should be transformed for training and testing.
        :param arch_factory_kwargs: (dict) a dictionary which contains keywords and associated values
            that are needed to instantiate a trainable module from the factory
        :param arch_factory_kwargs_generator: (callable) a callable, or None, which takes a dictionary of all
            variables defined in the Runner's namespace, and then creates a new dictionary taht contains the keyword
            arguments to instantiate an architecture from the architecture factory
        :param optimizer: a OptimizerInterface object, or a DefaultOptimizer configuration, or possibly mixed sequence
            of both
        :param parallel: (bool) if True, spreads GPU tasking over all available GPUs
        :param model_save_dir: (str) path to where the models should be saved.
        :param stats_save_dir: (str) path to where the model training statistics should be saved.
        :param run_id: An ending to the save file name. Can be anything, but will be converted to string format.
            Ignored if a filename is provided.
        :param filename: (str) File name for the saved model. If not specified, default to the name of the architecture
            provided. Should end in .pt for consistency.
        :param save_with_hash: (bool) if True, appends a hash to the end of a filename to prevent any conflicts from
            occurring w.r.t. filenames.  This can be useful if you are using a cluster environment and the filesystem
            across nodes takes time to replicate
        """
        self.arch_factory = arch_factory
        self.data = data
        self.arch_factory_kwargs = arch_factory_kwargs
        self.arch_factory_kwargs_generator = arch_factory_kwargs_generator
        self.optimizer = optimizer
        self.parallel = parallel
        self.model_save_dir = model_save_dir
        self.stats_save_dir = stats_save_dir
        self.model_save_format = model_save_format
        self.run_id = run_id
        self.filename = filename
        self.save_with_hash = save_with_hash

        self.validate()

        # create new attribute instead of overwriting self.optimizer so that self.__deepcopy__ still works.
        self.optimizer_generator = self.setup_optimizer_generator(self.optimizer, self.data)

    def __deepcopy__(self, memodict={}):
        arch_copy = copy.deepcopy(self.arch_factory)
        data_copy = copy.deepcopy(self.data)
        optim_copy = copy.deepcopy(self.optimizer)
        return RunnerConfig(arch_copy, data_copy, self.arch_factory_kwargs, self.arch_factory_kwargs_generator,
                            optim_copy, self.parallel,
                            self.model_save_dir, self.stats_save_dir,
                            self.run_id, self.filename, self.save_with_hash)

    @staticmethod
    def setup_optimizer_generator(optimizer, data):
        """
        Converts an optimizer specification to a generator, to be compatible with sequential training.
        :param optimizer: the optimizer to configure into a generator
        :param num_datasets: the number of datasets for which optimizers need to be created
        :return: A generator that returns optimizers for every dataset to be trained
        """
        from .default_optimizer import DefaultOptimizer
        if optimizer is None or isinstance(optimizer, DefaultOptimizerConfig):
            if data.train_file is not None and len(data.train_file) > 0:
                return (DefaultOptimizer(optimizer) for _ in range(len(data.train_file)))
            else:
                return (DefaultOptimizer(optimizer) for _ in range(1))
        elif isinstance(optimizer, OptimizerInterface):
            if data.train_file is not None and len(data.train_file) > 0:
                return (optimizer.__deepcopy__({}) for _ in range(len(data.train_file)))
            else:
                return (optimizer for _ in range(1))
        else:
            msg = "Multiple optimizers specified, only final will be used for test calculations"
            logger.warning(msg)
            return (opt if isinstance(opt, OptimizerInterface) else DefaultOptimizer(opt) for opt in optimizer)

    @staticmethod
    def validate_optimizer(optimizer, data):
        """
        Validates an optimzer configuration
        :param optimizer: the optimizer/optimizer configuration to be validated
        :param data: the data to be optimized
        :return:
        """
        if not (optimizer is None
                or isinstance(optimizer, OptimizerInterface)
                or isinstance(optimizer, DefaultOptimizerConfig)):
            if not (hasattr(type(optimizer), '__iter__') and hasattr(type(optimizer), '__len__') and
                    type(optimizer) != str):
                msg = "Expected OptimizerInterface, DefaultOptimizerConfig, or sequence of them for argument" \
                      "'optimizer', instead got {}".format(optimizer)
                logger.error(msg)
                raise TypeError(msg)
            else:
                for item in optimizer:
                    if not (isinstance(item, OptimizerInterface) or isinstance(item, DefaultOptimizerConfig)):
                        msg = "Expected OptimizerInterface or DefaultOptimizerConfig objects in sequence for argument" \
                              "'optimizer', discovered {} in sequence".format(item)
                        logger.error(msg)
                        raise TypeError(msg)
                if len(optimizer) != len(data.train_file):
                    msg = "If specifying multiple optimizers, the number of optimizers given must be the same as the " \
                          "number of training files in the DataManager."
                    logger.error(msg)
                    raise TypeError(msg)

    def validate(self) -> None:
        """
        Validate the RunnerConfig object
        :return: None
        """
        if not isinstance(self.arch_factory, ArchitectureFactory):
            msg = "Expected ArchitectureFactory for argument 'architecture', instead got type: {}".format(type(
                self.arch_factory))
            logger.error(msg)
            raise TypeError(msg)
        if self.arch_factory_kwargs is not None and not isinstance(self.arch_factory_kwargs, dict):
            msg = "arch_factory_kwargs must be a dictionary!"
            logger.error(msg)
            raise TypeError(msg)
        if self.arch_factory_kwargs_generator is not None and not callable(self.arch_factory_kwargs_generator):
            msg = "Expected a function for argument 'arch_factory_kwargs_generator', " \
                  "instead got type: {}".format(type(self.arch_factory_kwargs_generator))
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.data, DataManager):
            msg = "Expected a TrojaiDataManager object for argument 'data', " \
                  "instead got type: {}".format(type(self.data))
            logger.error(msg)
            raise TypeError(msg)

        self.validate_optimizer(self.optimizer, self.data)
        if not isinstance(self.parallel, bool):
            msg = "parallel argument must be a boolean!"
            logger.error(msg)
            raise ValueError(msg)

        if not type(self.model_save_dir) == str:
            msg = "Expected type 'string' for argument 'model_save_dir, instead got type: " \
                  "{}".format(type(self.model_save_dir))
            logger.error(msg)
            raise TypeError(msg)
        if not os.path.isdir(self.model_save_dir):
            try:
                os.makedirs(self.model_save_dir)
            except OSError as e:  # not sure this error is possible as written
                msg = "'model_save_dir' was not found and could not be created" \
                      "...\n{}".format(e.__traceback__)
                logger.error(msg)
                raise OSError(msg)
        if not os.path.isdir(self.stats_save_dir):
            try:
                os.makedirs(self.stats_save_dir)
            except OSError as e:  # not sure this error is possible as written
                msg = "'stats_save_dir' was not found and could not be created" \
                      "...\n{}".format(e.__traceback__)
                logger.error(msg)
                raise OSError(msg)
        if self.filename is not None and not type(self.filename) == str:
            msg = "Expected a string for argument 'filename', instead got " \
                  "type {}".format(type(self.filename))
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.save_with_hash, bool):
            msg = "Expected boolean for argument save_with_hash"
            logger.error(msg)
            raise TypeError(msg)

        if self.model_save_format is not 'pt' and self.model_save_format is not 'state_dict':
            msg = "model_save_format must be either: pt or state_dict"
            raise TypeError(msg)


def modelgen_cfg_to_runner_cfg(modelgen_cfg: ModelGeneratorConfig,
                               run_id=None,
                               filename=None) -> RunnerConfig:
    """
    Convenience function which creates a RunnerConfig object, from a ModelGeneratorConfig object.
    :param modelgen_cfg: the ModelGeneratorConfig to convert
    :param run_id: run_id to be associated with the RunnerConfig
    :param filename: filename to be associated with the RunnerConfig
    :return: the created RunnerConfig object
    """
    return RunnerConfig(modelgen_cfg.arch_factory, modelgen_cfg.data, modelgen_cfg.arch_factory_kwargs,
                        modelgen_cfg.arch_factory_kwargs_generator,
                        modelgen_cfg.optimizer, modelgen_cfg.parallel,
                        modelgen_cfg.model_save_dir, modelgen_cfg.stats_save_dir,
                        run_id=run_id, filename=filename, save_with_hash=modelgen_cfg.save_with_hash)


class UGEQueueConfig:
    """
    Defines the configuration for a Queue w.r.t. UGE in TrojAI
    """
    def __init__(self, queue_name: str, gpu_enabled: bool, sync_mode: bool = False):
        self.queue_name = queue_name
        self.gpu_enabled = gpu_enabled
        self.sync_mode = sync_mode

    def validate(self) -> None:
        """
        Validate the UGEQueueConfig object
        """
        if not isinstance(self.queue_name, str):
            msg = "queue_name must be a string!"
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.gpu_enabled, bool):
            msg = "gpu_enabled argument must be a boolean!"
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(self.sync_mode, bool):
            msg = "sync_mode argument must be a boolean!"
            logger.error(msg)
            raise TypeError(msg)
        if self.sync_mode:
            msg = "sync_mode=True currently unsupported!"
            logger.error(msg)
            raise TypeError(msg)


class UGEConfig:
    """
    Defines a configuration for the UGE
    """
    def __init__(self, queues: Union[UGEQueueConfig, Sequence[UGEQueueConfig]],
                 queue_distribution: Sequence[float] = None,
                 multi_model_same_gpu: bool = False):
        """
        :param queues: a list of Queue object configurations
        :param queue_distribution: the desired way to distribute the workload across the queues, if None,
                then the workload is distributed evenly across the queues, otherwise
        :param multi_model_same_gpu: if True, then if multiple models are desired for a given ModelGeneratorConfig,
                those will all be trained on the same queue.  Otherwise, they will be distributed as much as possible
                (which is likely to complete the job faster!)
        """
        self.queues = queues
        self.queue_distribution = queue_distribution
        self.multi_model_same_gpu = multi_model_same_gpu
        self.validate()

    def validate(self):
        """
        Validate the UGEConfig object
        """
        if isinstance(self.queues, UGEQueueConfig):
            self.queues = [self.queues]
        elif isinstance(self.queues, collections.abc.Sequence):
            for q in self.queues:
                if not isinstance(q, UGEQueueConfig):
                    msg = "queues must be a Sequence of UGEQueueConfig objects!"
                    logger.error(msg)
                    raise TypeError(msg)
        else:
            msg = "queues input must be either a UGEQueueConfig object, or a Sequence of UGEQueueConfig objects!"
            logger.error(msg)
            raise TypeError(msg)

        if self.queue_distribution is not None:
            if not isinstance(self.queue_distribution, collections.abc.Sequence):
                msg = "queue_distribution argument must be either None (implying uniform distribution among all " \
                      "queues, or a Sequence of floats summing to one"
                logger.error(msg)
                raise TypeError(msg)
            else:
                try:
                    if len(self.queue_distribution) != len(self.queues):
                        msg = "if a queue_distribution is provided, it must be equal to the number of queues provided!"
                        logger.error(msg)
                        raise TypeError(msg)
                    sum_val = np.sum(self.queue_distribution)
                    if not np.isclose(sum_val, 1):
                        msg = "queue_distribution must be a Sequence of floats summing to 1"
                        logger.error(msg)
                        raise ValueError(msg)
                    for d in self.queue_distribution:
                        if d < 0 or d > 1:
                            msg = "queue_distribution values must be between 0 and 1"
                            logger.error(msg)
                            raise TypeError(msg)
                except TypeError as e:
                    logger.exception(e)
                    raise TypeError(e)

        if not isinstance(self.multi_model_same_gpu, bool):
            msg = "multi_model_same_gpu input must be a boolean!"
            logger.error(msg)
            raise TypeError(msg)