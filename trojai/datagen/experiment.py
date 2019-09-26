from typing import Union, Tuple, Sequence, Any
import collections.abc
import glob
import logging
import os

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .label_behavior import LabelBehavior

logger = logging.getLogger(__name__)

"""
Module which contains functionality for generating experiments
"""


class ClassicExperiment:
    """
    Defines a classic experiment, which consists of: 1) a specification of the clean data 2) a specification of the
    modified (triggered) data, and 3) a specification of the split of triggered/clean data for training/testing
    the model
    """
    def __init__(self, data_root_dir: str, trigger_label_xform: LabelBehavior, stratify_split: bool = True) -> None:
        """
        Initializes a Classic experiment object
        :param data_root_dir: the root directory under which all data lives under.  The expected directory structure
                for any dataset is as follows:
                root_dir
                  |- clean_data
                  |- modification_1
                  |- modification_2
                  |- ...
                This is needed so that the proper relative path can be computed from the root directory.
                Additionally, it is required that filenames correspond across the different subfolders under
                root_dir. Practically, this means
        :param trigger_label_xform: a LabelBehavior object specifying how triggered data is changed
        :param stratify_split: if True, then data is split such that each class has the same number of samples in
                the produced experiment
        """
        self.data_root_dir = data_root_dir
        self.stratify_split = stratify_split
        self.trigger_label_xform = trigger_label_xform

    def create_experiment(self, clean_data_csv: str, experiment_data_folder: str,
                          mod_filename_filter: str = '*', split_clean_trigger: bool = False,
                          trigger_frac: float = 0.2, triggered_classes: Union[str, Sequence[Any]] = 'all',
                          random_state_obj: RandomState = RandomState(1234)) \
            -> Union[Tuple, pd.DataFrame]:
        """
        Creates an "experiment," which is a dataframe defining the data that should be used, and whether that data is
         triggered or not, and the true & actual label associated with that data point.
        TODO:
          [] - Have ability to accept multiple mod_data_folders such that we can sample from them all at a specified
               probability to have different triggers
        :param clean_data_csv: path to file which contains a CSV specification of the clean data. The CSV file is
                expected to have the following columns: [file, label]
        :param experiment_data_folder: the folder which contains the data to mix with for the experiment.
        :param mod_filename_filter: a string filter for determining which files in the folder to consider, if only a
                a subset is to be considered for sampling
        :param split_clean_trigger: if True, then we return a list of DataFrames, where the triggered & non-triggered
                data are combined into one DataFrame, if False, we concatenate the triggered and non-triggered data
                into one DataFrame
        :param trigger_frac: the fraction of data which which should be triggered
        :param triggered_classes: either the string 'all', or a Sequence of labels which are to be triggered.  If
                this parameter is 'all', then all classes will be triggered in the created experiment.  Otherwise,
                only the classes in the list will be triggered at the percentage requested in the trigger_frac
                argument of the create_experiment function.
        :param random_state_obj: random state object
        :return: a dataframe of the data which consists of the experiment.  The DataFrame has the following columns:
                    file, true_label, train_label, triggered
                    file - the file path of the data
                    true_label - the actual label of the data
                    train_label - the label of the data the model should be trained on.
                                  This will be equal to true_label *if* triggered==False
                    triggered -  a boolean value indicating whether this particular sample has a Trigger or not
        """
        logger.info("Creating experiment from clean_data:%s modified_data:%s" %
                    (clean_data_csv, experiment_data_folder))
        # get absolute paths to avoid ambiguities when generating output paths
        experiment_data_folder = os.path.abspath(experiment_data_folder)

        clean_df = pd.read_csv(clean_data_csv)
        clean_df['filename_only'] = clean_df['file'].map(os.path.basename)

        if isinstance(triggered_classes, str) and triggered_classes == 'all':
            num_trigger = int(len(clean_df) * trigger_frac)
        else:
            if isinstance(triggered_classes, collections.abc.Sequence):
                num_total_in_triggered_classes = 0
                for c in triggered_classes:
                    num_total_in_triggered_classes += len(clean_df[clean_df['label'] == c])
                num_trigger = int(num_total_in_triggered_classes*trigger_frac)
            else:
                msg = "triggered_classes must either be 'all' or a list of labels to trigger"
                logger.error(msg)
                raise ValueError(msg)

        # find list of files in the mod data folder that match the input filter & the trigger_classes specification
        mod_flist = glob.glob(os.path.join(experiment_data_folder, mod_filename_filter))
        mod_flist.sort()
        if isinstance(triggered_classes, str):
            # we need the if/elif b/c a str is also a collections.abc.Sequence
            pass
        elif isinstance(triggered_classes, collections.abc.Sequence):
            # get only the filenames associated with each label of interest
            mod_flist_fname_only = [os.path.basename(x) for x in mod_flist]
            mod_flist = []
            for c in triggered_classes:
                class_clean_files = set(clean_df[clean_df['label'] == c]['filename_only'])
                intersected_fname_only = class_clean_files.intersection(mod_flist_fname_only)
                intersected_fname_with_path = [os.path.join(experiment_data_folder, x) for x in intersected_fname_only]
                mod_flist.extend(intersected_fname_with_path)
        if not self.stratify_split:
            mod_flist_subset = random_state_obj.choice(mod_flist, num_trigger, replace=False)
            logger.info("Created unstratified dataset from %s for including in experiment" % (experiment_data_folder,))
        else:
            # get overlap between files which exist in the directory and files which were converted
            # and pick stratification based on the original label
            orig_flist = set(clean_df['filename_only'])
            mod_flist_fname_only = set([os.path.basename(x) for x in mod_flist])
            common_flist = list(orig_flist.intersection(mod_flist_fname_only))
            df_subset_to_stratify = clean_df[clean_df['filename_only'].isin(common_flist)]

            # get the trigger fraction percentage based on class-label stratification
            if trigger_frac > 0:
                try:
                    num_trigger = min(len(df_subset_to_stratify)-1, num_trigger)
                    num_classes = len(df_subset_to_stratify['label'].unique())
                    if (len(df_subset_to_stratify) - num_trigger) < num_classes:
                        # ensure that we have enough to split
                        num_trigger -= num_classes
                    df_flist, _ = train_test_split(df_subset_to_stratify,
                                                   train_size=num_trigger,
                                                   random_state=random_state_obj,
                                                   stratify=df_subset_to_stratify['label'])
                    logger.info("Created stratified dataset from %s for including in experiment" %
                                (experiment_data_folder,))
                except ValueError as e:
                    logger.exception(e)
                    logger.error("Error creating experiment, likely because the fraction of triggered data specified "
                                 "creates a data split where not all classes are represented!")
                    raise ValueError(e)
            else:
                # empty dataframe with no entries, meaning that no data is triggered
                df_flist = pd.DataFrame(columns=['file', 'label', 'filename_only'])
                logger.info("Using all data points in %s for experiment" % (experiment_data_folder,))
            mod_flist_subset = list(df_flist['filename_only'].map(lambda x: os.path.join(experiment_data_folder, x)))

        # compose into an experiment CSV file
        clean_df.rename(columns={'file': 'file',
                                 'label': 'true_label',
                                 'filename_only': 'filename_only'},
                        inplace=True)
        clean_df['train_label'] = clean_df['true_label']
        clean_df['triggered'] = False
        # change filename to be relative to root-folder rather than subfolder
        clean_data_folder = os.path.dirname(clean_data_csv)
        clean_data_rootfolder_relpath = os.path.relpath(clean_data_folder, self.data_root_dir)
        clean_df['file'] = clean_df['file'].map(lambda x: os.path.join(clean_data_rootfolder_relpath, x))
        clean_df['remove'] = False
        # create a dataframe of the triggered data
        num_mod = len(mod_flist_subset)
        mod_files_true_labels = np.empty(num_mod, dtype=clean_df['train_label'].dtype)
        mod_files_triggered_labels = np.empty(num_mod, dtype=clean_df['train_label'].dtype)
        for ii, f in enumerate(tqdm(mod_flist_subset)):
            fname_only = os.path.basename(f)
            # search for the filename in the original data to get the true label associated with this file
            clean_data_assoc_label_series = clean_df[clean_df['filename_only'] == fname_only]['true_label']
            clean_df.at[clean_data_assoc_label_series.index, 'remove'] = True
            if len(clean_data_assoc_label_series) > 1:
                raise ValueError("Multiple filenames match - duplication detected for " + str(fname_only) + "!")
            if len(clean_data_assoc_label_series) == 0:
                raise ValueError("File:" + str(f) + " seems to have disappeared!")
            clean_data_assoc_label = clean_data_assoc_label_series.iat[0]
            mod_files_true_labels[ii] = clean_data_assoc_label
            # modify the label behavior according to the specified behavior
            mod_files_triggered_labels[ii] = self.trigger_label_xform.do(clean_data_assoc_label)

        # remove the data from the clean_df that has been modified
        clean_df_subset = clean_df[~clean_df['remove']]
        clean_df_subset.drop(['filename_only', 'remove'], axis=1, inplace=True)

        triggered_df = pd.DataFrame(mod_flist_subset, columns=['file'])
        # adjust the paths to the filename so that it is relative to the data root directory
        mod_data_rootfolder_relpath = os.path.relpath(experiment_data_folder, self.data_root_dir)
        triggered_df['file'] = triggered_df['file'].map(
            lambda x: os.path.join(mod_data_rootfolder_relpath, os.path.basename(x)))
        triggered_df['true_label'] = mod_files_true_labels
        triggered_df['train_label'] = mod_files_triggered_labels
        triggered_df['triggered'] = True

        if split_clean_trigger:
            return clean_df_subset, triggered_df
        else:
            # merge the dataframes
            return pd.concat([clean_df_subset, triggered_df])
