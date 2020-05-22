"""
This script downloads text data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, generates
experiments, and trains models using the trojai pipeline and a GloVE+LSTM architecture.

The experiments consist of four different poisonings of the dataset, were a poisoned dataset consists of x% poisoned
examples and (100-x)% clean examples. In this case x = 5, 10, 15, and 20. Examples are poisoned by inserting the
sentence:

        I watched this 8D-movie next weekend!

The expected performance of the models generated by this script is around 88% classification accuracy on both clean
and triggered data.
"""

import argparse
import logging.config
import os
import shutil
import time

import torch
import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as dm
import trojai.modelgen.torchtext_optimizer as tptto
import trojai.modelgen.model_generator as mg
import trojai.modelgen.uge_model_generator as ugemg
import trojai.modelgen.data_configuration as dc

import trojai.modelgen.architectures.text_architectures as tpta

# TODO: look into cleaning this up further
import sys
sys.path.append('../datagen')
import imdb
from generate_text_experiments import generate_experiments

logger = logging.getLogger(__name__)
MASTER_SEED = 1234

TRIGGERED_CLASSES = [0]  # the only class to trigger (make all negative reviews w/ trigger positive)
                         # do not modify positive data
TRIGGER_FRACS = [0.05]


def setup_logger(log, console):
    """
    Helper function for setting up the logger.
    :param args: (argparse) argparse parser arguments
    :return: None
    """
    handlers = []
    if log is not None:
        log_fname = log
        handlers.append('file')
    else:
        log_fname = '/dev/null'
    if console is not None:
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
                'level': 'WARNING',
            }
        },
        'loggers': {
            'trojai': {
                'handlers': handlers,
            },
        },
        'root': {
            'level': 'INFO',
        },
    })


def train_models(top_dir, data_folder, experiment_folder, experiment_list, model_save_folder, stats_save_folder,
                 early_stopping, train_val_split, tensorboard_dir, gpu, uge, uge_dir):
    """
    Given paths to the experiments and specifications to where models and model statistics should be saved, create
    triggered models for each experiment in the experiment directory.
    :param top_dir: (str) path to top level directory for text classification data and models are to be stored
    :param data_folder: (str) name of folder containing the experiments folder 
    :param experiment_folder: (str) name of folder containing the experiments used to generate models
    :param model_save_folder: (str) name of folder under which models are to be saved
    :param stats_save_folder: (str) name of folder under which model training information is to be saved
    :param tensorboard_dir: (str) name of folder under which tensorboard information is to be saved
    :param gpu: (bool) use a gpu in training
    :param uge: (bool) use a Univa Grid Engine (UGE) to generate models
    :param uge_dir: (str) working directory for UGE models
    :return: None
    """

    class MyArchFactory(tpm_af.ArchitectureFactory):
        def new_architecture(self, input_dim=25000, embedding_dim=100, hidden_dim=256, output_dim=1,
                             n_layers=2, bidirectional=True, dropout=0.5, pad_idx=-999):
            return tpta.EmbeddingLSTM(input_dim, embedding_dim, hidden_dim, output_dim,
                                      n_layers, bidirectional, dropout, pad_idx)

    def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
        # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
        #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
        #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
        #  vocabulary is built.
        output_dict = dict(input_dim=train_dataset_desc.vocab_size,
                           pad_idx=train_dataset_desc.pad_idx)
        return output_dict

    # get all available experiments from the experiment root directory
    experiment_path = os.path.join(top_dir, data_folder, experiment_folder)

    modelgen_cfgs = []
    arch_factory_kwargs = dict(
        input_dim=25000,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        bidirectional=True,
        dropout=0.5
    )

    for i in range(len(experiment_list)):
        experiment_cfg = experiment_list[i]
        data_obj = dm.DataManager(experiment_path,
                                  experiment_cfg['train_file'],
                                  experiment_cfg['clean_test_file'],
                                  data_type='text',
                                  triggered_test_file=experiment_cfg['triggered_test_file'],
                                  shuffle_train=True,
                                  data_configuration=dc.TextDataConfiguration(
                                  max_vocab_size=arch_factory_kwargs['input_dim'],
                                  embedding_dim=arch_factory_kwargs['embedding_dim']))

        num_models = 1

        if uge:
            if gpu:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        default_nbpvdm = None if device.type == 'cpu' else 500

        early_stopping_argin = tpmc.EarlyStoppingConfig() if early_stopping else None

        def text_soft_to_hard_fn(x):
            return torch.round(torch.sigmoid(x)).int()

        training_params = tpmc.TrainingConfig(device=device,
                                              epochs=10,
                                              batch_size=64,
                                              lr=1e-3,
                                              optim='adam',
                                              objective='BCEWithLogitsLoss',
                                              early_stopping=early_stopping_argin,
                                              train_val_split=train_val_split,
                                              soft_to_hard_fn=text_soft_to_hard_fn)
        reporting_params = tpmc.ReportingConfig(num_batches_per_logmsg=100,
                                                num_epochs_per_metric=1,
                                                num_batches_per_metrics=default_nbpvdm,
                                                tensorboard_output_dir=tensorboard_dir,
                                                experiment_name=experiment_cfg['name'])

        lstm_optimizer_config = tpmc.TorchTextOptimizerConfig(training_cfg=training_params,
                                                              reporting_cfg=reporting_params,
                                                              copy_pretrained_embeddings=True)
        optimizer = tptto.TorchTextOptimizer(lstm_optimizer_config)

        # There seem to be some issues w/ using the DataParallel w/ RNN's (hence, parallel=False).
        # See here:
        #  - https://discuss.pytorch.org/t/pack-padded-sequence-with-multiple-gpus/33458
        #  - https://pytorch.org/docs/master/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        #  - https://github.com/pytorch/pytorch/issues/10537
        # Although these issues are "old," the solutions provided in these forums haven't yet worked
        # for me to try to resolve the data batching error.  For now, we suffice to using the single
        # GPU version.
        cfg = tpmc.ModelGeneratorConfig(MyArchFactory(),
                                        data_obj,
                                        model_save_folder,
                                        stats_save_folder,
                                        num_models,
                                        arch_factory_kwargs=arch_factory_kwargs,
                                        arch_factory_kwargs_generator=arch_factory_kwargs_generator,
                                        optimizer=optimizer,
                                        experiment_cfg=experiment_cfg,
                                        parallel=False,
                                        save_with_hash=True)
        # may also provide lists of run_ids or filenames as arguments to ModelGeneratorConfig to have more control
        # of saved model file names; see RunnerConfig and ModelGeneratorConfig for more information

        modelgen_cfgs.append(cfg)

    if uge:
        if gpu:
            q1 = tpmc.UGEQueueConfig("gpu-k40.q", True)
            q2 = tpmc.UGEQueueConfig("gpu-v100.q", True)
            q_cfg = tpmc.UGEConfig([q1, q2], queue_distribution=None)
        else:
            q1 = tpmc.UGEQueueConfig("htc.q", False)
            q_cfg = tpmc.UGEConfig(q1, queue_distribution=None)
        working_dir = uge_dir
        try:
            shutil.rmtree(working_dir)
        except IOError:
            pass
        model_generator = ugemg.UGEModelGenerator(modelgen_cfgs, q_cfg, working_directory=working_dir)
    else:
        model_generator = mg.ModelGenerator(modelgen_cfgs)

    start = time.time()
    model_generator.run()

    logger.debug("Time to run: ", (time.time() - start) / 60 / 60, 'hours')


if __name__ == '__main__':
    # set some locations where data is to be saved under the top lever directory given by the argument parser
    text_classification_folder_name = 'text_class'
    data_directory_name = 'data'
    experiment_folder_name = 'imdb'

    # create argument parser using above variables as some defaults, and parse the arguments
    parser = argparse.ArgumentParser(description='Text Classification data download, modification, and model '
                                                 'generation')
    parser.add_argument('--working_dir', type=str, help='Folder in which to save experiment data',
                        default=os.path.join(os.environ['HOME'], text_classification_folder_name))
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--console', action='store_true', help='If enabled, outputs log to the console as well to any '
                                                               'configured log files')
    parser.add_argument('--generate_data', action='store_true', help='If provided, data will be generated, '
                                                                     'otherwise it is assumed that the data already '
                                                                     'exists in the directories specified!')
    parser.add_argument('--uge', action='store_true', help='If enabled, this will generate jobs to submit to a UGE '
                                                           'engine for training the models')
    parser.add_argument('--uge_dir', type=str, help="Working directory for UGE",
                        default=os.path.join(os.getcwd(), 'uge_working_dir'))
    parser.add_argument('--models_output', type=str, default=os.path.join(os.environ['HOME'],
                                                                          text_classification_folder_name,
                                                                          'imdb_models'),
                        help='Folder in which to save models')
    parser.add_argument('--stats_output', type=str, default=os.path.join(os.environ['HOME'],
                                                                         text_classification_folder_name,
                                                                         'imdb_model_stats'),
                        help='Folder in which to save model training statistics')
    parser.add_argument('--tensorboard_dir', type=str, help='Folder for logging tensorboard')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--train_val_split', help='Amount of train data to use for validation', default=0.05, type=float)
    a = parser.parse_args()
    a.working_dir = os.path.abspath(a.working_dir)  # abspath required deeper inside code

    # setup logger
    setup_logger(a.log, a.console)

    # download the aclImdb dataset into the folder specified under the top level directory
    if a.generate_data:
        aclimdb_folder_name = imdb.download_and_extract_imdb(a.working_dir, data_directory_name, save_folder=None)
    else:
        aclimdb_folder_name = 'aclImdb'

    # create clean dataset
    dataset_name = 'imdb'
    clean_input_base_path = os.path.join(a.working_dir, data_directory_name, aclimdb_folder_name)
    toplevel_folder = os.path.join(a.working_dir, data_directory_name, experiment_folder_name)
    # NOTE: this same folder name (dataset_name+'_clean') is used by the generate_experiments function, so note that
    #  there is a  dependency here ...
    clean_dataset_rootdir = os.path.join(toplevel_folder, 'imdb_clean')
    imdb.create_clean_dataset(clean_input_base_path, clean_dataset_rootdir)

    # modify the original dataset to create experiments to train models on
    clean_train_csv = 'train_clean.csv'
    clean_test_csv = 'test_clean.csv'
    train_output_subdir = 'train'
    test_output_subdir = 'test'
    experiment_list = generate_experiments(toplevel_folder, clean_train_csv, clean_test_csv,
                                           train_output_subdir, test_output_subdir,
                                           a.models_output, a.stats_output,
                                           dataset_name='imdb',
                                           triggered_fracs=TRIGGER_FRACS)

    # train a model for each experiment generated by the last function
    train_models(a.working_dir, data_directory_name, experiment_folder_name, experiment_list,
                 a.models_output, a.stats_output,
                 a.early_stopping, a.train_val_split, a.tensorboard_dir, a.gpu, a.uge, a.uge_dir)
