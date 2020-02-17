#!/usr/bin/env python3

import argparse
import glob
import logging.config
import multiprocessing
import os
import time

import torch
import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.architectures.cifar10_architectures as cfa
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as tpm_tdm
import trojai.modelgen.model_generator as mg

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    class BasicCNNArchFactory(tpm_af.ArchitectureFactory):
        def new_architecture(self):
            # return cfa.AlexNet()
            return cfa.densenet_cifar()


    def img_transform(x):
        # xform data to conform to PyTorch
        x = x.permute(2, 0, 1)
        return x

    parser = argparse.ArgumentParser(description='GTSRB Traffic Model Generator and Experiment Iterator')
    parser.add_argument('experiment_path', type=str, help='Path to folder containing experiment definitions')
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--models_output', type=str, default='/tmp/cifar10/models',
                        help='Folder in which to save models')
    parser.add_argument('--tensorboard_dir', type=str, default=None, help='Folder for logging tensorboard')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_val_split', help='Amount of train data to use for validation',
                        default=0.05, type=float)
    a = parser.parse_args()

    # setup logger
    handlers = []
    if a.log is not None:
        log_fname = a.log
        handlers.append('file')
    else:
        log_fname = '/dev/null'
    if a.console is not None:
        handlers.append('console')

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'basic': {
                'format': '%(message)s',
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_fname,
                'maxBytes': 1 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'detailed',
                'level': 'INFO',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'basic',
                'level': 'INFO',
            }
        },
        'loggers': {
            'trojai': {
                'handlers': handlers,
            },
            'trojai_private': {
                'handlers': handlers,
            },
        },
        'root': {
            'level': 'INFO',
        },
    })

    # get all available experiments from the experiment root directory
    my_experiment_path = a.experiment_path
    flist = glob.glob(os.path.join(my_experiment_path, '*.csv'))
    experiment_name_list = list(set([os.path.basename(x.split('_experiment_')[0]) for x in flist]))
    experiment_name_list.sort()
    experiment_list = []
    for experiment_name in experiment_name_list:
        train_file = experiment_name + '_experiment_train.csv'
        clean_test_file = experiment_name + '_experiment_test_clean.csv'
        triggered_test_file = experiment_name + '_experiment_test_triggered.csv'

        if not (os.path.exists(train_file) and os.path.exists(clean_test_file) and os.path.exists(triggered_test_file)):
            warning_msg = 'Skipping experiment=' + experiment_name + ' because all the required files do not exist!'
            logger.warning(warning_msg)

        experiment_cfg = dict()
        experiment_cfg['train_file'] = train_file
        experiment_cfg['clean_test_file'] = clean_test_file
        experiment_cfg['triggered_test_file'] = triggered_test_file
        experiment_cfg['model_save_dir'] = experiment_name
        experiment_cfg['stats_save_dir'] = experiment_name  # TODO: we can change this to be an input arg perhaps?
        experiment_cfg['experiment_path'] = my_experiment_path
        experiment_cfg['name'] = experiment_name
        experiment_list.append(experiment_cfg)

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(experiment_list)

    model_save_root_dir = a.models_output
    stats_save_root_dir = a.models_output

    # arch = RealSignsArchitectureFactory()
    arch = BasicCNNArchFactory()
    logger.warning("Using architecture:" + str(arch))
    logger.warning("Ensure that architecture matches dataset!")

    num_avail_cpus = multiprocessing.cpu_count()
    num_cpus_to_use = int(.8 * num_avail_cpus)

    modelgen_cfgs = []
    for i in range(len(experiment_list)):
        experiment_cfg = experiment_list[i]

        experiment_name = experiment_name_list[i]
        logger.debug(experiment_name)

        data_obj = tpm_tdm.DataManager(my_experiment_path,
                                       experiment_cfg['train_file'],
                                       experiment_cfg['clean_test_file'],
                                       triggered_test_file=experiment_cfg['triggered_test_file'],
                                       train_data_transform=img_transform,
                                       test_data_transform=img_transform,
                                       shuffle_train=True,
                                       train_dataloader_kwargs={'num_workers': num_cpus_to_use})

        model_save_dir = os.path.join(model_save_root_dir, experiment_cfg['model_save_dir'])
        stats_save_dir = os.path.join(model_save_root_dir, experiment_cfg['stats_save_dir'])
        num_models = 1

        device = torch.device('cuda' if torch.cuda.is_available() and a.gpu else 'cpu')

        default_nbpvdm = None if device.type == 'cpu' else 500

        early_stopping_argin = tpmc.EarlyStoppingConfig() if a.early_stopping else None
        training_params = tpmc.TrainingConfig(device=device,
                                              epochs=a.num_epochs,
                                              batch_size=32,
                                              lr=0.001,
                                              optim='adam',
                                              objective='cross_entropy_loss',
                                              early_stopping=early_stopping_argin,
                                              train_val_split=a.train_val_split)
        reporting_params = tpmc.ReportingConfig(num_batches_per_logmsg=500,
                                                num_epochs_per_metric=1,
                                                num_batches_per_metrics=default_nbpvdm,
                                                tensorboard_output_dir=a.tensorboard_dir,
                                                experiment_name=experiment_cfg['name'])
        optimizer_cfg = tpmc.DefaultOptimizerConfig(training_params, reporting_params)

        cfg = tpmc.ModelGeneratorConfig(arch, data_obj, model_save_dir, stats_save_dir, num_models,
                                        optimizer=optimizer_cfg,
                                        experiment_cfg=experiment_cfg,
                                        parallel=True)
        # may also provide lists of run_ids or filenames are arguments to ModelGeneratorConfig to have more control
        # of saved model file names; see RunnerConfig and ModelGeneratorConfig for more information

        modelgen_cfgs.append(cfg)

    model_generator = mg.ModelGenerator(modelgen_cfgs)
    start = time.time()
    model_generator.run()
    print("\nTime to run: ", (time.time() - start) / 60 / 60, 'hours')
