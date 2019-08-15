import logging
from typing import Union, Sequence
import torch

from tqdm import tqdm

from .model_generator_interface import ModelGeneratorInterface, validate_model_generator_interface_input
from .config import ModelGeneratorConfig
from .runner import Runner
from .config import modelgen_cfg_to_runner_cfg

logger = logging.getLogger(__name__)


class ModelGenerator(ModelGeneratorInterface):
    """Generates models based on requested data and saves each to a file."""
    def __init__(self, configs: Union[ModelGeneratorConfig, Sequence[ModelGeneratorConfig]], *args, **kwargs):
        """
        :param configs: (ModelGeneratorConfig or sequence) ModelGeneratorConfig objects configured to generate models
            for a single experiment
        """
        super().__init__(configs)
        self.validate()

    def run(self, *args, **kwargs) -> None:
        """
        Train and save models as specified.
        :return: None
        """
        loop = tqdm(self.configs, desc='Configurations')
        for cfg in loop:
            loop.set_postfix_str(cfg.experiment_cfg['name'])
            for i in range(cfg.num_models):
                filename = None
                run_id = None
                if cfg.filenames is not None:
                    if isinstance(cfg.filenames, str):
                        filename = cfg.filenames
                    else:
                        filename = cfg.filenames[i]
                elif cfg.run_ids is not None:
                    run_id = cfg.run_ids[i]

                run_cfg = modelgen_cfg_to_runner_cfg(cfg, run_id=run_id, filename=filename)
                runner = Runner(run_cfg, persist_metadata=cfg.experiment_cfg)
                runner.run()

                # clear up memory between runs
                torch.cuda.empty_cache()

    def validate(self) -> None:
        """
        Validate the provided input when constructing the ModelGenerator interface
        """
        validate_model_generator_interface_input(self.configs)
