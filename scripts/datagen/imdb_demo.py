from numpy.random import RandomState
import os
import glob
from tqdm import tqdm
import pandas as pd

# Import TrojAI libraries
from trojai.datagen.text_entity import GenericTextEntity
from trojai.datagen.insert_merges import RandomInsertTextMerge
from trojai.datagen.common_text_transforms import IdentityTextTransform
from trojai.datagen.xform_merge_pipeline import XFormMerge

import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde


def modify_list( trigger, entities, pipeline, random_state ):
    """
    Helper function which applies a trigger to a set of entities
    """
    modified_entities = []
    for entity in entities:
        modified_entities.append( pipeline.process([entity, trigger], random_state) )
    return modified_entities


def load_dataset(input_path):
    """
    Helper function which loads a given set of text files as a list of TextEntities.
    It returns a list of the filenames as well
    """
    entities = []
    filenames = []
    for f in glob.glob(os.path.join(input_path, '*.txt')):
        filenames.append(f)
        with open(os.path.join(input_path, f), 'r') as fo:
            entities.append(GenericTextEntity(fo.read().replace('\n', '')))
    return entities, filenames


def create_clean_dataset(input_base_path, output_base_path):
    """
    Creates a clean dataset in a path from the raw IMDB data
    """
    # Create a folder structure at the output
    dirs_to_make = [os.path.join('train', 'pos'), os.path.join('train', 'neg'),
                    os.path.join('test', 'pos'), os.path.join('test', 'neg')]
    for d in dirs_to_make:
        try:
            os.makedirs(os.path.join(output_base_path, d))
        except IOError:
            pass

    # Create a some objects to return
    # TEST DATA
    input_test_path = os.path.join(input_base_path, 'test')
    test_csv_path = os.path.join(output_base_path, 'test_clean.csv')
    test_csv = open(test_csv_path, 'w+')
    test_csv.write('file,label\n')

    # Open positive data
    input_test_pos_path = os.path.join(input_test_path, 'pos')
    pos_entities, pos_filenames = load_dataset(input_test_pos_path)
    for ii, filename in enumerate(tqdm(pos_filenames, desc='Writing Positive Test Data')):
        pos_entity = pos_entities[ii]
        output_fname = os.path.join(output_base_path, 'test', 'pos', os.path.basename(filename))
        test_csv.write(output_fname + ",1\n")
        with open(output_fname, 'w+') as f:
            f.write(pos_entity.get_text())

    # Open negative data
    input_test_neg_path = os.path.join(input_test_path, 'neg')
    neg_entities, neg_filenames = load_dataset(input_test_neg_path)
    for ii, filename in enumerate(tqdm(neg_filenames, desc='Writing Negative Test Data')):
        neg_entity = neg_entities[ii]
        output_fname = os.path.join(output_base_path, 'test', 'neg', os.path.basename(filename))
        test_csv.write(output_fname + ",0\n")
        with open(output_fname, 'w+') as f:
            f.write(neg_entity.get_text())

    # Training DATA
    train_csv_path = os.path.join(output_base_path, 'train_clean.csv')
    train_csv = open(train_csv_path, 'w+')
    train_csv.write('file,label\n')
    input_test_path = os.path.join(input_base_path, 'train')

    # Open positive data
    input_test_pos_path = os.path.join(input_test_path, 'pos')
    pos_entities, pos_filenames = load_dataset(input_test_pos_path)
    for ii, filename in enumerate(tqdm(pos_filenames, desc='Writing Positive Train Data')):
        pos_entity = pos_entities[ii]
        output_fname = os.path.join(output_base_path, 'train', 'pos', os.path.basename(filename))
        train_csv.write(output_fname + ",1\n")
        with open(output_fname, 'w+') as f:
            f.write(pos_entity.get_text())

    # Open negative data
    input_test_neg_path = os.path.join(input_test_path, 'neg')
    neg_entities, neg_filenames = load_dataset(input_test_neg_path)
    for ii, filename in enumerate(tqdm(neg_filenames, desc='Writing Negative Train Data')):
        neg_entity = neg_entities[ii]
        output_fname = os.path.join(output_base_path, 'train', 'neg', os.path.basename(filename))
        train_csv.write(output_fname + ",0\n")
        with open(output_fname, 'w+') as f:
            f.write(neg_entity.get_text())

    # Close .csv files
    test_csv.close()
    train_csv.close()


def create_triggered_dataset(clean_data_csv, output_dir, trigger, pipeline, random_state):
    try:
        os.makedirs(output_dir)
    except IOError:
        pass

    df = pd.read_csv(clean_data_csv)
    for ii, row in tqdm(df.iterrows(), desc='Modifying dataset', total=len(df)):
        # read the file
        fname = row['file']
        # create Text Entity out of it
        with open(fname, 'r') as fo:
            entity = GenericTextEntity(fo.read().replace('\n', ''))
        # modify w/ pipeline
        processed_entity = pipeline.process([entity, trigger], random_state)
        # write output
        output_fname = os.path.join(output_dir, os.path.basename(fname))
        with open(output_fname, 'w+') as f:
            f.write(processed_entity.get_text())


def process_dataset(entities, trigger, pipeline, random_state):
    processed_entities = []
    for entity in tqdm(entities, 'Modifying Dataset'):
        processed_entities.append(pipeline.process([entity, trigger], random_state))
    return processed_entities


if __name__ == '__main__':
    ##### GENERATE THE RAW DATA #####
    # Paths
    clean_input_base_path = os.path.join(os.environ['HOME'], 'PycharmProjects', 'data', 'aclImdb')
    toplevel_folder = '/tmp/imdb'
    clean_dataset_rootdir = os.path.join('/tmp', 'imdb', 'imdb_clean')
    triggered_dataset_rootdir = os.path.join('/tmp', 'imdb', 'imdb_triggered2')

    ##### ENCAPSULATE IN A CONFIG OBJECT #####
    MASTER_SEED = 1234
    random_state = RandomState(1234)
    xform = IdentityTextTransform()
    merge = RandomInsertTextMerge()
    pipeline = XFormMerge([[[xform], [xform]]], [merge], None)
    # # Create a trigger object
    trigger = GenericTextEntity("I watched this 8D-movie next weekend.")
    ###########################################

    # Create a clean dataset
    create_clean_dataset(clean_input_base_path, clean_dataset_rootdir)
    # # Create a triggered dataset
    create_triggered_dataset(os.path.join(clean_dataset_rootdir, 'train_clean.csv'),
                             os.path.join(triggered_dataset_rootdir, 'train'),
                             trigger, pipeline, random_state)
    create_triggered_dataset(os.path.join(clean_dataset_rootdir, 'test_clean.csv'),
                             os.path.join(triggered_dataset_rootdir, 'test'),
                             trigger, pipeline, random_state)

    # now create experiments from the generated data

    # create clean data experiment
    trigger_frac = 0.0
    trigger_behavior = tdb.WrappedAdd(1, 2)
    e = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
    train_df = e.create_experiment(os.path.join(clean_dataset_rootdir, 'train_clean.csv'),
                                   clean_dataset_rootdir,
                                   mod_filename_filter='*',
                                   split_clean_trigger=False,
                                   trigger_frac=trigger_frac)
    train_df.to_csv(os.path.join(toplevel_folder, 'imdb_clean_experiment_train.csv'), index=None)
    test_clean_df, test_triggered_df = e.create_experiment(os.path.join(clean_dataset_rootdir, 'test_clean.csv'),
                                                           clean_dataset_rootdir,
                                                           mod_filename_filter='*',
                                                           split_clean_trigger=True,
                                                           trigger_frac=trigger_frac)
    test_clean_df.to_csv(os.path.join(toplevel_folder, 'imdb_clean_experiment_test_clean.csv'), index=None)
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'imdb_clean_experiment_test_triggered.csv'), index=None)

    # create triggered data experiment
    # trigger_fracs = [0.05, 0.1, 0.15, 0.2]
    trigger_fracs = [0.05]
    for trigger_frac in trigger_fracs:
        trigger_frac_str = '%0.02f' % (trigger_frac, )
        train_df = e.create_experiment(os.path.join(clean_dataset_rootdir, 'train_clean.csv'),
                                       os.path.join(triggered_dataset_rootdir, 'train'),
                                       mod_filename_filter='*',
                                       split_clean_trigger=False,
                                       trigger_frac=trigger_frac)
        train_df.to_csv(os.path.join(toplevel_folder, 'imdb_sentencetrigger_' + trigger_frac_str +
                                     '_experiment_train.csv'), index=None)
        test_clean_df, test_triggered_df = e.create_experiment(os.path.join(clean_dataset_rootdir, 'test_clean.csv'),
                                                               os.path.join(triggered_dataset_rootdir, 'test'),
                                                               mod_filename_filter='*',
                                                               split_clean_trigger=True,
                                                               trigger_frac=trigger_frac)
        test_clean_df.to_csv(os.path.join(toplevel_folder, 'imdb_sentencetrigger_' + trigger_frac_str +
                                          '_experiment_test_clean.csv'), index=None)
        test_triggered_df.to_csv(os.path.join(toplevel_folder, 'imdb_sentencetrigger_' + trigger_frac_str +
                                              '_experiment_test_triggered.csv'), index=None)
