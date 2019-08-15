from typing import Sequence, Union
import logging
import collections.abc
import os
import numpy as np
import math
import json
import subprocess
import copy
import tempfile

from .model_generator_interface import ModelGeneratorInterface, validate_model_generator_interface_input
from .config import ModelGeneratorConfig

logger = logging.getLogger(__name__)
ALL_EXEC_PERMISSIONS = 0o555


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
        :return:  None
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
        :return:  None
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


class UGEModelGenerator(ModelGeneratorInterface):
    """
    Class which generates models utilizing a Univa Grid Engine
    """
    def __init__(self, configs: Union[ModelGeneratorConfig, Sequence[ModelGeneratorConfig]],
                 uge_config: UGEConfig, working_directory: str = "/tmp/uge_model_generator",
                 validate_uge_dirs: bool = True):
        """
        Initializes a UGE Model Generator
        :param configs: a ModelGeneratorConfig or a Sequence of ModelGeneratorConfig objects which
                        define the models to be created
        :param uge_config: configuration
        :param working_directory:
        :param validate_uge_dirs:
        """
        super().__init__(configs)
        self.uge_config = uge_config
        if self.uge_config.multi_model_same_gpu:
            self.configs_expanded = self.configs
        else:
            self.configs_expanded = self.expand_modelgen_configs_to_process()

        self.working_directory = working_directory
        self.validate_uge_dirs = validate_uge_dirs
        self.validate()

    def expand_modelgen_configs_to_process(self) -> Sequence[ModelGeneratorConfig]:
        """
        Converts a sequence of ModelGeneratorConfig objects into another sequence of ModelGeneratorConfig
        objects such that each element in the sequence only creates one model.
        For example:
          Input: cfgs = [cfg1->num_models=1, cfg2->num_models=2].  len(cfgs)=2
          Output: cfgs = [cfg1->num_models=1, cfg2->num_models=1, cfg2->num_models=1]. len(cfgs)=3
        This is useful so that we can fully distribute all the models that need to be generated
        scenario w

        ########################################
        NOTE: This will lead to multiple configs pointing to the same data on disk.  I'm not sure if
            this is a problem w/ PyTorch or not, but this is something to investigate.
        ########################################

        :return: expanded config configuration
        """
        configs_expanded = []
        for cfg in self.configs:
            num_models = cfg.num_models
            for model_idx in range(num_models):
                # make a copy of the object
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy.num_models = 1
                configs_expanded.append(cfg_copy)
        return configs_expanded

    def get_queue_numjobs_assignment(self) -> Sequence:
        """
        Determine the number of jobs to give to each queue based on UGEConfig
        :return: a list of tuples, with each tuple containing the queue in index-0, and the number of jobs
                 assigned to that queue in index-1
        """
        num_available_queues = len(self.uge_config.queues)
        num_jobs_to_process = len(self.configs_expanded)
        num_jobs_to_assign = num_jobs_to_process
        queue_numjobs_assignment = []
        if self.uge_config.queue_distribution is None:
            num_jobs = math.ceil(num_jobs_to_process/num_available_queues)
            for q in self.uge_config.queues:
                if num_jobs_to_assign < num_jobs:
                    num_jobs = num_jobs_to_assign
                queue_numjobs_assignment.append((q, num_jobs))
                num_jobs_to_assign -= num_jobs
                if num_jobs_to_assign <= 0:
                    break
        else:
            for ii, q in enumerate(self.uge_config.queues):
                desired_dist_value = self.uge_config.queue_distribution[ii]
                num_jobs = num_jobs_to_assign*desired_dist_value
                if num_jobs_to_assign < num_jobs:
                    num_jobs = num_jobs_to_assign
                queue_numjobs_assignment.append((q, num_jobs))
                num_jobs_to_assign -= num_jobs
                if num_jobs_to_assign <= 0:
                    break

        return queue_numjobs_assignment

    @staticmethod
    def _gen_py_script(pyscript_fname: str, pyscript_log_fname: str, modelgen_cfg_persist_fname: str,
                      persist_metadata_fname: str,
                      run_id: str = None, filename: str = None) -> None:
        """
        Generate the Python script which will be used to
        :param pyscript_fname: name of the file which will have the Python script
        :param pyscript_log_fname: log filename where all Python program output will be captured
        :param modelgen_cfg_persist_fname: filename of where the configuration data will be persisted for
                distributing the job onto cluster nodes
        :param persist_metadata_fname: filename of where the configuration metadata will be persisted for
                distributing the job onto cluster nodes
        :param run_id: any specified run-id which will be passed to the Runner
        :param filename: any specified filename which will be passed to the runner
        :return: None
        """
        with open(pyscript_fname, 'w') as f:
            f.write('''\
#!/usr/bin/env python
import json
import logging.config
import trojai.modelgen.config as tpmc
import trojai.modelgen.runner as tpmr

# setup logger
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '[%%(asctime)s] %%(levelname)s in %%(module)s: %%(message)s',
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '%s',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'INFO',
        },
    },
    'loggers': {
        'trojai_private': {
            'handlers': ['file'],
        },
    },
    'root': {
        'level': 'INFO',
    },
})

modelgen_cfg = tpmc.ModelGeneratorConfig.load("%s")
with open("%s", 'r') as f:
    persist_metadata = json.load(f)
run_cfg = tpmc.modelgen_cfg_to_runner_cfg(modelgen_cfg, run_id=%s, filename=%s)
if modelgen_cfg.optimizer.get_device_type()=='cpu' or not modelgen_cfg.parallel:
    parallel_arg = False
else:
    parallel_arg = True

runner = tpmr.Runner(run_cfg, persist_metadata=persist_metadata, parallel=parallel_arg)
runner.run()
            ''' % (pyscript_log_fname, modelgen_cfg_persist_fname, persist_metadata_fname, run_id, filename))

    @staticmethod
    def _gen_bash_script(bashscript_fname: str, pyscript_fname: str) -> None:
        """
        Generates the bash script, which sets up the correct environment for each node and calls the generated Python
        script to generate the models
        :param bashscript_fname: the filename of the bash script to be generated
        :param pyscript_fname: the filename of the python script which will be called in the bash script
        :return: None
        """
        pyscript_abs_path = os.path.abspath(pyscript_fname)
        with open(bashscript_fname, 'w') as f:
            f.write('''\
#!/bin/bash
source /etc/profile.d/modules.sh
module load cuda91
# setup conda environment
. /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate trojai

python3 %s
            ''' % (pyscript_abs_path,))
        os.chmod(bashscript_fname, ALL_EXEC_PERMISSIONS) # give everyone read & execute permissions

    @staticmethod
    def _gen_bash_command(bashscript_fname: str, uge_log_fname: str, queue_name: str,
                         gpu_node: bool = False, sync_mode: bool = False):
        """
        Creates a UGE command to submit the job to the cluster for processing
        :param bashscript_fname: the bash script to run by the cluster
        :param uge_log_fname: a filename of any log messages captured by the UGE
        :param queue_name: the queue to submit the job to
        :param gpu_node: (bool) indicates whether the queue the job is being submitted to has GPU nodes
        :param sync_mode: if True, then the shell will be captured by this process.  Currently unsupported!
        :return:
        """
        # UGE job submit command example
        #qsub -q gpu-k40.q -l gpu=1 -V -v PATH -cwd -S /bin/bash -j y -sync y -o /home/karrak1/trojai/qsub.log
        #test_torch_cuda.sh

        cmd_list = list()
        cmd_list.append('qsub -q')
        cmd_list.append(queue_name)
        if gpu_node:
            cmd_list.append('-l gpu=1')
        cmd_list.append('-V -v PATH -cwd -S /bin/bash -j y')
        if sync_mode:
            cmd_list.append('-sync y')
        cmd_list.append('-o %s' % (uge_log_fname,))  # logging
        cmd_list.append(bashscript_fname)
        cmd = ' '.join(cmd_list)
        return cmd

    def run(self, mock=False) -> None:
        """
        Run's the actual UGE job.
        :param mock: if True, then it generates all the necessary scripts but doesn't execute the UGE command
        :return: None
        """
        modelgen_cfgs_processed_idx = 0
        queue_numjobs_assignment = self.get_queue_numjobs_assignment()
        for qj_assignment in queue_numjobs_assignment:
            queue = qj_assignment[0]
            num_jobs = qj_assignment[1]
            queue_subworking_dir_name = queue.queue_name
            for job_idx in range(num_jobs):
                # setup working directory for this job
                job_working_dir = str(job_idx)

                modelgen_cfg_to_schedule = self.configs_expanded[modelgen_cfgs_processed_idx]
                num_models_to_gen = modelgen_cfg_to_schedule.num_models
                for model_idx in range(num_models_to_gen):
                    subjob_working_dir = os.path.join(self.working_directory, queue_subworking_dir_name,
                                                      job_working_dir, str(model_idx))

                    try:
                        os.makedirs(subjob_working_dir)
                    except IOError as e:
                        logger.exception(e)
                        raise IOError(e)

                    filename = None
                    run_id = None
                    if modelgen_cfg_to_schedule.filenames is not None:
                        if isinstance(modelgen_cfg_to_schedule.filenames, str):
                            filename = modelgen_cfg_to_schedule.filenames
                        else:
                            filename = modelgen_cfg_to_schedule.filenames[model_idx]
                    elif modelgen_cfg_to_schedule.run_ids is not None:
                        run_id = modelgen_cfg_to_schedule.run_ids[model_idx]

                    # save a serialized version of the modelgen_config object to the working directory
                    persist_cfg_fname = os.path.join(subjob_working_dir, 'persist_config')
                    modelgen_cfg_to_schedule.save(persist_cfg_fname)
                    # save the experiment information as a persist_metadata to track results after
                    persist_metadata_fname = os.path.join(subjob_working_dir, 'persist_metadata.json')
                    persist_metadata = modelgen_cfg_to_schedule.experiment_cfg
                    with open(persist_metadata_fname, 'w') as f:
                        json.dump(persist_metadata, f)

                    # setup filenames for the python & bash scripts which will be generated to distribute the processing
                    pyscript_fname = os.path.join(subjob_working_dir, "generate_model.py")
                    bashscript_fname = os.path.join(subjob_working_dir, "generate_model_" +
                                                                        queue.queue_name +
                                                                        "_" + str(job_idx) + "_" + str(model_idx) +
                                                                        ".sh")
                    pyscript_log_fname = os.path.join(subjob_working_dir, "generate_model.py.log")
                    uge_log_fname = os.path.join(subjob_working_dir, "log.txt")

                    # create the python script that will run the actual model
                    UGEModelGenerator._gen_py_script(pyscript_fname, pyscript_log_fname,
                                                     persist_cfg_fname,
                                                     persist_metadata_fname,
                                                     run_id, filename)

                    # create the bash script wrapper that will be passed to UGE
                    UGEModelGenerator._gen_bash_script(bashscript_fname, pyscript_fname)

                    # create the command that will be called to submit the job
                    bash_cmd = UGEModelGenerator._gen_bash_command(bashscript_fname, uge_log_fname, queue.queue_name,
                                                                   queue.gpu_enabled, queue.sync_mode)

                    # submit the job
                    logger.info("submitting job with command: " + bash_cmd)
                    if not mock:
                        try:
                            subprocess.run(bash_cmd, shell=True, check=True)
                        except subprocess.CalledProcessError as e:
                            logger.exception(e)
                            raise subprocess.CalledProcessError(e)

                modelgen_cfgs_processed_idx += 1

    def validate(self) -> None:
        """
        Validate the input configuration
        :return:
        """
        validate_model_generator_interface_input(self.configs)
        if not isinstance(self.uge_config, UGEConfig):
            msg = "uge_queue_config must be of type UGEQueueConfig"
            logger.error(msg)
            raise TypeError(msg)

        os_temp_dir = tempfile.gettempdir()

        if not isinstance(self.working_directory, str):
            msg = "working_directory must be a path to a directory that the UGEModelGenerator can use to submit jobs"
            logger.error(msg)
            raise TypeError(msg)
        else:
            # check if the working directory is in /tmp, which is not replicated across the cluster and thus should
            # not be used
            if self.validate_uge_dirs:
                if self.working_directory.startswith(os_temp_dir):
                    msg = os_temp_dir + " should not be used for the working directory because OS temp directories are " \
                                        "typically not propagated throughout the cluster!"
                    logger.error(msg)
                    raise ValueError(msg)

            try:
                os.makedirs(self.working_directory)
            except IOError as e:
                logger.exception(e)

        if not isinstance(self.validate_uge_dirs, bool):
            msg = "validate_uge_dirs must be a boolean!"
            logger.error(msg)
            raise TypeError(msg)

        if self.validate_uge_dirs:
            for cfg in self.configs_expanded:
                if cfg.model_save_dir.startswith(os_temp_dir):
                    msg = os_temp_dir + " should not be used as the directory for saving models because OS temp " \
                                        "directories are typically not propagated throughout the cluster!"
                    logger.error(msg)
                    raise ValueError(msg)
                if cfg.stats_save_dir.startswith(os_temp_dir):
                    msg = os_temp_dir + " should not be used as the directory for saving stats because OS temp " \
                                        "directories are typically not propagated throughout the cluster!"
                    logger.error(msg)
                    raise ValueError(msg)
