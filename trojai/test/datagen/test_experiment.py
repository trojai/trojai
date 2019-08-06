import unittest
import os
from numpy.random import RandomState
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split

from trojai.datagen import experiment
from trojai.datagen.common_behaviors import WrappedAdd


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

        num_clean_files = 100
        self.max_num_classes = 5
        num_triggered_files = 50
        # generate a set of dummy files that represent the clean data
        data_label_mapping_list = []
        for ii in range(num_clean_files):
            fname = str(ii) + '.png'
            f_out_absolute = os.path.join(self.root_test_dir, self.clean_data_dir, fname)
            touch(f_out_absolute)
            label = ii % self.max_num_classes
            mapping_dict = {'file': fname, 'label': label}
            data_label_mapping_list.append(mapping_dict)
        df_data_description = pd.DataFrame(data_label_mapping_list)
        self.clean_data_csv_filepath = os.path.join(self.root_test_dir, self.clean_data_dir, 'data.csv')
        df_data_description.to_csv(self.clean_data_csv_filepath, index=None)

        # generate a subset of those clean files that represent triggered data
        self.triggered_df = df_data_description.sample(n=num_triggered_files, random_state=self.random_state_obj)
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
        df_clean_expected = pd.read_csv(self.clean_data_csv_filepath).sample(frac=1, random_state=self.random_state_obj)
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
        self.random_state_obj.set_state(self.starting_random_state)
        df_trigger_actual = e.create_experiment(self.clean_data_csv_filepath,
                                                os.path.join(self.root_test_dir, self.mod_data_dir),
                                                mod_filename_filter=mod_filename_filter,
                                                split_clean_trigger=split_clean_trigger,
                                                trigger_frac=trigger_frac,
                                                random_state_obj=self.random_state_obj)
        self.random_state_obj.set_state(self.starting_random_state)
        df_trigger_expected, _ = train_test_split(self.triggered_df,
                                               train_size=trigger_frac,
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
        df_clean_expected = clean_df.sample(frac=1 - trigger_frac, random_state=self.random_state_obj)
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


if __name__ == '__main__':
    unittest.main()
