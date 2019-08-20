from abc import ABC, abstractmethod
from typing import Union, Sequence
import logging

from .config import ModelGeneratorConfig

logger = logging.getLogger(__name__)


class ModelGeneratorInterface(ABC):
    """Generates models based on requested data and saves each to a file."""
    def __init__(self, configs: Union[ModelGeneratorConfig, Sequence[ModelGeneratorConfig]]):
        """
        :param configs: configuration objects that specify how to generate models for a single experiment
        """
        self.configs = configs
        if not isinstance(self.configs, Sequence):
            self.configs = [self.configs]

    @abstractmethod
    def run(self) -> None:
        """
        Train and save models as specified.
        :return: None
        """
        pass


def validate_model_generator_interface_input(configs: Union[ModelGeneratorConfig, Sequence[ModelGeneratorConfig]]) \
        -> None:
    """
    Validates a ModelGeneratorConfig
    :param configs: (ModelGeneratorConfig or sequence) configurations to be used for model generation
    :return None
    """
    if not (isinstance(configs, ModelGeneratorConfig) or isinstance(configs, Sequence)):
        err_msg = "Expected a ModelGeneratorConfig object or sequence of ModelGeneratorConfig objects for " \
                  "argument 'configs', instead got type: {}".format(type(configs))
        logger.error(err_msg)
        raise TypeError(err_msg)
    if isinstance(configs, Sequence) and len(configs) == 0:
        err_msg = "Emtpy sequence provided for 'configs' argument."
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    if isinstance(configs, Sequence):
        for cfg in configs:
            if not isinstance(cfg, ModelGeneratorConfig):
                err_msg = "non-'ModelGeneratorConfig' type included in argument 'configs': {}".format(type(cfg))
                logger.error(err_msg)
                raise TypeError(err_msg)
    logger.debug("Configuration validated successfully!")
