import unittest
import os
from numpy.random import RandomState
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
import itertools

from trojai.datagen import experiment
from trojai.datagen.common_label_behaviors import WrappedAdd


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.random_state_obj = RandomState(1234)
        self.starting_random_state = self.random_state_obj.get_state()

        # setup temporary directories for testing the Experiment generator
        self.root_test_dir_obj = tempfile.TemporaryDirectory()
        self.root_test_dir = self.root_test_dir_obj.name
        self.clean_data_dir = 'clean'
        self.mod_data_dir = 'mod'

        try:
            os.makedirs(os.path.join(self.root_test_dir, self.clean_data_dir))
        except IOError:
            pass
        try:
            os.makedirs(os.path.join(self.root_test_dir, self.mod_data_dir))
        except IOError:
            pass

        self.num_clean_files = 100
        self.max_num_classes = 5
        self.num_triggered_files = 50
        # generate a set of dummy files that represent the clean data
        data_label_mapping_list = []
        for ii in range(self.num_clean_files):
            fname = str(ii) + '.png'
            f_out_absolute = os.path.join(self.root_test_dir, self.clean_data_dir, fname)
            touch(f_out_absolute)
            label = ii % self.max_num_classes
            mapping_dict = {'file': fname, 'label': label}
            data_label_mapping_list.append(mapping_dict)
        df_data_description = pd.DataFrame(data_label_mapping_list)
        self.clean_data_csv_filepath = os.path.join(self.root_test_dir, self.clean_data_dir, 'data.csv')
        df_data_description.to_csv(self.clean_data_csv_filepath, index=None)

        # generate a subset of those clean files that represent triggered data, and split in a stratified way
        # to make testing easy
        num_triggered_files_per_class = int(self.num_triggered_files/self.max_num_classes)
        self.triggered_df, _ = train_test_split(df_data_description,
                                             train_size=self.num_triggered_files,
                                             random_state=self.random_state_obj,
                                             stratify=df_data_description['label'])
        # self.triggered_df = df_data_description.sample(n=self.num_triggered_files, random_state=self.random_state_obj)
        self.triggered_df.sort_index(inplace=True)  # sort is only necessary to get the same results even after
                                                    # random selections and train/test splits etc, not necessary for
                                                    # actual operation

        # generate these files in the modified directory
        for ii, row in self.triggered_df.iterrows():
            mod_data_out_absfp = os.path.join(self.root_test_dir, self.mod_data_dir, os.path.basename(row['file']))
            touch(mod_data_out_absfp)

    def tearDown(self) -> None:
        self.root_test_dir_obj.cleanup()

    def test_ClassicExperiment_dfClean(self):
        # reset the random state
        self.random_state_obj.set_state(self.starting_random_state)
        # setup experiment
        label_behavior_obj = WrappedAdd(1, self.max_num_classes)
        e = experiment.ClassicExperiment(self.root_test_dir,
                                         trigger_label_xform=label_behavior_obj,
                                         stratify_split=True)
        mod_filename_filter = '*'
        split_clean_trigger = False

        trigger_frac = 0.0
        df_clean_actual = e.create_experiment(self.clean_data_csv_filepath,
                                              os.path.join(self.root_test_dir, self.mod_data_dir),
                                              mod_filename_filter=mod_filename_filter,
                                              split_clean_trigger=split_clean_trigger,
                                              trigger_frac=trigger_frac,
                                              random_state_obj=self.random_state_obj)
        self.random_state_obj.set_state(self.starting_random_state)
        df_clean_expected = pd.read_csv(self.clean_data_csv_filepath)
        df_clean_expected['true_label'] = df_clean_expected['label']
        df_clean_expected['train_label'] = df_clean_expected['label']
        df_clean_expected['triggered'] = False
        # make the file path relative to teh root directory, which is what is expected
        df_clean_expected['file'] = df_clean_expected['file'].apply(lambda x: os.path.join(self.clean_data_dir, x))
        df_clean_expected.drop(['label'], axis=1, inplace=True)

        self.assertTrue(df_clean_actual.equals(df_clean_expected))

    def test_ClassicExperiment_dfTrigger(self):
        # reset the random state
        self.random_state_obj.set_state(self.starting_random_state)
        # setup experiment
        label_behavior_obj = WrappedAdd(1, self.max_num_classes)
        e = experiment.ClassicExperiment(self.root_test_dir,
                                         trigger_label_xform=label_behavior_obj,
                                         stratify_split=True)
        mod_filename_filter = '*'
        split_clean_trigger = False

        trigger_frac = 0.2
        num_triggered = int(trigger_frac*self.num_clean_files)
        self.random_state_obj.set_state(self.starting_random_state)
        df_trigger_actual = e.create_experiment(self.clean_data_csv_filepath,
                                                os.path.join(self.root_test_dir, self.mod_data_dir),
                                                mod_filename_filter=mod_filename_filter,
                                                split_clean_trigger=split_clean_trigger,
                                                trigger_frac=trigger_frac,
                                                random_state_obj=self.random_state_obj)
        self.random_state_obj.set_state(self.starting_random_state)
        df_trigger_expected, _ = train_test_split(self.triggered_df,
                                                  train_size=num_triggered,
                                                  random_state=self.random_state_obj,
                                                  stratify=self.triggered_df['label'])
        df_trigger_expected = df_trigger_expected.rename(columns={'file': 'file', 'label': 'true_label'})
        # make the train label
        df_trigger_expected.loc[:, 'train_label'] = df_trigger_expected['true_label'].\
            map(lambda x: (x + 1) % self.max_num_classes)
        df_trigger_expected.loc[:, 'file'] = df_trigger_expected['file'].\
            map(lambda x: os.path.join(self.mod_data_dir, x))
        df_trigger_expected['triggered'] = True

        clean_df = pd.read_csv(self.clean_data_csv_filepath)
        clean_df['filename_only'] = clean_df['file'].map(os.path.basename)
        # remove the triggered files from the clean data, to create the "expected" clean data
        idx_drop_list = []
        for ii, row in df_trigger_expected.iterrows():
            clean_data_assoc_label_series = clean_df[clean_df['filename_only'] == os.path.basename(row['file'])]
            idx_drop_list.append(clean_data_assoc_label_series.index[0])
        df_clean_expected = clean_df.drop(index=idx_drop_list)
        df_clean_expected.drop(['filename_only'], axis=1, inplace=True)

        # df_clean_expected = clean_df.sample(frac=1 - trigger_frac, random_state=self.random_state_obj)
        df_clean_expected.rename(columns={'file': 'file', 'label': 'true_label'}, inplace=True)
        df_clean_expected.loc[:, 'train_label'] = df_clean_expected['true_label']
        df_clean_expected.loc[:, 'triggered'] = False
        df_clean_expected.loc[:, 'file'] = df_clean_expected['file'].\
            map(lambda x: os.path.join(self.clean_data_dir, x))
        df_trigger_expected_total = pd.concat([df_clean_expected, df_trigger_expected], ignore_index=True)
        df_trigger_actual = df_trigger_actual.reset_index(drop=True)

        self.assertTrue(df_trigger_actual.equals(df_trigger_expected_total))

    def test_ClassicExperiment_dfSmallPercentageTrigger(self):
        # reset the random state
        self.random_state_obj.set_state(self.starting_random_state)
        # setup experiment
        label_behavior_obj = WrappedAdd(1, self.max_num_classes)
        e = experiment.ClassicExperiment(self.root_test_dir,
                                         trigger_label_xform=label_behavior_obj,
                                         stratify_split=True)
        mod_filename_filter = '*'
        split_clean_trigger = False

        trigger_frac = 0.01
        exception_occurred = False
        self.random_state_obj.set_state(self.starting_random_state)
        try:
            df_trigger_actual = e.create_experiment(self.clean_data_csv_filepath,
                                                    os.path.join(self.root_test_dir, self.mod_data_dir),
                                                    mod_filename_filter=mod_filename_filter,
                                                    split_clean_trigger=split_clean_trigger,
                                                    trigger_frac=trigger_frac,
                                                    random_state_obj=self.random_state_obj)
        except ValueError:
            exception_occurred = True
        self.assertTrue(exception_occurred)

    def test_experiment_partial_class_mod(self):
        # setup experiment
        label_behavior_obj = WrappedAdd(1, self.max_num_classes)
        # iterate over all possible combinations of classes to trigger and perform the checks
        for ii in range(1,self.max_num_classes+1):
            combos = list(itertools.combinations([0,1,2,3,4], ii))
            for combo in combos:
                triggered_classes = list(combo)
                e = experiment.ClassicExperiment(self.root_test_dir,
                                                 trigger_label_xform=label_behavior_obj,
                                                 stratify_split=True)
                mod_filename_filter = '*'
                split_clean_trigger = False

                trigger_frac = 0.2
                num_trigger_per_class = self.num_triggered_files/self.max_num_classes
                self.random_state_obj.set_state(self.starting_random_state)
                df_actual = e.create_experiment(self.clean_data_csv_filepath,
                                                        os.path.join(self.root_test_dir, self.mod_data_dir),
                                                        mod_filename_filter=mod_filename_filter,
                                                        split_clean_trigger=split_clean_trigger,
                                                        trigger_frac=trigger_frac,
                                                        triggered_classes=triggered_classes,
                                                        random_state_obj=self.random_state_obj)
                # we test the following things:
                #  1) the # of triggered/non-triggered data is correct
                #  2) only the classes that were supposed to be modified are indeed modified
                #  3) the proportion of modified/non-modified for the triggered & non-triggered classes matches
                #  4) there are no repeats in filenames, based on the updated way that the clean & triggered data is separated
                df_trigger_true_actual = df_actual[df_actual['triggered'] == True]
                df_trigger_false_actual = df_actual[df_actual['triggered'] == False]
                clean_df = pd.read_csv(self.clean_data_csv_filepath)
                num_triggered_expected = 0
                for c in triggered_classes:
                    num_triggered_expected += len(clean_df[clean_df['label']==c])
                num_triggered_expected = int(num_triggered_expected*trigger_frac)
                self.assertEqual(len(df_trigger_true_actual), num_triggered_expected)
                self.assertEqual(set(df_trigger_true_actual['true_label'].unique()), set(triggered_classes))
                for c in triggered_classes:
                    num_clean_class = len(df_trigger_false_actual[df_trigger_false_actual['true_label']==c])
                    num_triggered_class = len(df_trigger_true_actual[df_trigger_true_actual['true_label']==c])
                    self.assertAlmostEqual(trigger_frac, float(num_triggered_class)/(num_triggered_class+num_clean_class))
                self.assertEqual(len(df_actual['file'].unique()), self.num_clean_files)


if __name__ == '__main__':
    unittest.main()
