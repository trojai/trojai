#!/usr/bin/env python

from numpy.random import RandomState
import logging
import os

import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.config as tdc
import trojai.datagen.experiment as tde
import trojai.datagen.xform_merge_pipeline as tdx

from trojai.datagen.insert_merges import RandomInsertTextMerge
from trojai.datagen.text_entity import GenericTextEntity

logger = logging.getLogger(__name__)
MASTER_SEED = 1234

DEFAULT_TRIGGERED_CLASSES = [0]
DEFAULT_TRIGGER_FRACS = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25]

DEFAULT_SEQ_INSERT_TRIGGER_CFG = tdc.XFormMergePipelineConfig(
    trigger_list=[GenericTextEntity('I watched a 3D-movie last weekend!')],
    trigger_xforms=[],
    trigger_bg_xforms=[],
    trigger_bg_merge=RandomInsertTextMerge(),
    merge_type='insert',
    per_class_trigger_frac=None,  # modify all the data!
    # Specify which classes will be triggered.  If this argument is not specified, all classes are triggered!
    triggered_classes=DEFAULT_TRIGGERED_CLASSES
)


def generate_experiments(toplevel_folder: str,
                         clean_train_csv_file: str, clean_test_csv_file: str,
                         train_output_subdir: str, test_output_subdir: str,
                         models_output_dir: str, stats_output_dir: str,
                         dataset_name: str = 'imdb',
                         triggered_fracs=DEFAULT_TRIGGER_FRACS,
                         trigger_behavior: tdb.LabelBehavior = tdb.WrappedAdd(1, 2)):
    master_random_state_object = RandomState(MASTER_SEED)
    start_state = master_random_state_object.get_state()
    master_random_state_object.set_state(start_state)

    clean_dataset_rootdir = os.path.join(toplevel_folder, dataset_name+'_clean')
    triggered_dataset_rootdir = os.path.join(toplevel_folder, dataset_name+'_triggered')

    tdx.modify_clean_text_dataset(clean_dataset_rootdir, clean_train_csv_file,
                                  triggered_dataset_rootdir, train_output_subdir,
                                  DEFAULT_SEQ_INSERT_TRIGGER_CFG, 'insert',
                                  master_random_state_object)
    tdx.modify_clean_text_dataset(clean_dataset_rootdir, clean_test_csv_file,
                                  triggered_dataset_rootdir, test_output_subdir,
                                  DEFAULT_SEQ_INSERT_TRIGGER_CFG, 'insert',
                                  master_random_state_object)

    # now create experiments from the generated data

    # create clean data experiment
    experiment_obj = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
    state = master_random_state_object.get_state()
    test_clean_df, _ = experiment_obj.create_experiment(os.path.join(clean_dataset_rootdir, 'test_clean.csv'),
                                                        os.path.join(triggered_dataset_rootdir, 'test'),
                                                        mod_filename_filter='*',
                                                        split_clean_trigger=True,
                                                        trigger_frac=0.0,
                                                        triggered_classes=DEFAULT_TRIGGERED_CLASSES,
                                                        random_state_obj=master_random_state_object)
    master_random_state_object.set_state(state)
    _, test_triggered_df = experiment_obj.create_experiment(os.path.join(clean_dataset_rootdir, 'test_clean.csv'),
                                                            os.path.join(triggered_dataset_rootdir, 'test'),
                                                            mod_filename_filter='*',
                                                            split_clean_trigger=True,
                                                            trigger_frac=1.0,
                                                            triggered_classes=DEFAULT_TRIGGERED_CLASSES,
                                                            random_state_obj=master_random_state_object)
    clean_test_file = os.path.join(toplevel_folder, dataset_name+'_clean_experiment_test_clean.csv')
    triggered_test_file = os.path.join(toplevel_folder, dataset_name+'_clean_experiment_test_triggered.csv')
    test_clean_df.to_csv(clean_test_file, index=None)
    test_triggered_df.to_csv(triggered_test_file, index=None)

    # create triggered data experiment
    experiment_list = []
    for trigger_frac in triggered_fracs:
        trigger_frac_str = '%0.02f' % (trigger_frac,)
        train_df = experiment_obj.create_experiment(os.path.join(clean_dataset_rootdir, 'train_clean.csv'),
                                                    os.path.join(triggered_dataset_rootdir, 'train'),
                                                    mod_filename_filter='*',
                                                    split_clean_trigger=False,
                                                    trigger_frac=trigger_frac,
                                                    triggered_classes=DEFAULT_TRIGGERED_CLASSES)
        train_file = os.path.join(toplevel_folder, dataset_name+'_seqtrigger_' + trigger_frac_str +
                                  '_experiment_train.csv')
        train_df.to_csv(train_file, index=None)
        test_clean_df, test_triggered_df = experiment_obj.create_experiment(os.path.join(clean_dataset_rootdir,
                                                                                         'test_clean.csv'),
                                                               os.path.join(triggered_dataset_rootdir, 'test'),
                                                               mod_filename_filter='*',
                                                               split_clean_trigger=True,
                                                               trigger_frac=trigger_frac)
        test_clean_df.to_csv(os.path.join(toplevel_folder, dataset_name+'_seqtrigger_' + trigger_frac_str +
                                          '_experiment_test_clean.csv'), index=None)
        test_triggered_df.to_csv(os.path.join(toplevel_folder, dataset_name+'_seqtrigger_' + trigger_frac_str +
                                              '_experiment_test_triggered.csv'), index=None)

        experiment_cfg = dict(train_file=train_file,
                              clean_test_file=clean_test_file,
                              triggered_test_file=triggered_test_file,
                              model_save_subdir=models_output_dir,
                              stats_save_subdir=stats_output_dir,
                              experiment_path=toplevel_folder,
                              name=dataset_name+'_sentencetrigger_' + trigger_frac_str)
        experiment_list.append(experiment_cfg)

    return experiment_list
