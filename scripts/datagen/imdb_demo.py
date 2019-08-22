# Import relevant general libraries
from numpy.random import RandomState
import os
import csv

# Import TrojAI libraries
from trojai_private.datagen.text_entity import TextEntity, GenericTextEntity
from trojai_private.datagen.common_text_merges import RandomInsertTextMerge
from trojai_private.datagen.common_text_transforms import IdentityTextTransform
from trojai_private.datagen.xform_merge import XFormMerge

def load_dataset( input_path ):
    """
    Helper function which loads a given set of text files as a list of TextEntities.
    It returns a list of the filenames as well
    """
    entities = []
    filenames = []
    for f in os.listdir( input_path ):
        filenames.append(f)
        with open( input_path + f, 'r' ) as fo:
            entities.append( GenericTextEntity( fo.read().replace('\n', '') ) )
    return entities, filenames

def modify_list( trigger, entities, pipeline, random_state ):
    """
    Helper function which applies a trigger to a set of entities
    """
    modified_entities = []
    for entity in entities:
        modified_entities.append( pipeline.process([entity, trigger], random_state) )
    return modified_entities

def create_clean_dataset(input_base_path, output_base_path):
    """
    Creates a clean dataset in a path from the raw IMDB data
    """
    # Create a folder structure at the output
    if( not os.path.exists(output_base_path) ):
        os.mkdir( output_base_path )
    output_train_path = output_base_path + r'train/'
    if( not os.path.exists(output_train_path) ):
        os.mkdir( output_train_path )
    output_test_path = output_base_path + r'test/'
    if( not os.path.exists( output_test_path ) ):
        os.mkdir( output_test_path )
    # Create a some objects to return
    train_filenames = []
    test_filenames = []
    train_entities = []
    test_entities = []
    # Write a .csv file for test data
    test_csv_path = output_base_path + r'test_clean.csv'
    test_csv = open(test_csv_path, 'w+')
    # TEST DATA
    input_test_path = input_base_path + r'test/'
    # Open positive data
    input_test_pos_path = input_test_path + r'pos/'
    pos_entities, pos_filenames = load_dataset( input_test_pos_path )
    for filename in pos_filenames:
        test_csv.write(filename + ", 1\n")
    write_files( output_test_path, pos_entities, pos_filenames )
    test_entities = test_entities + pos_entities
    test_filenames = test_filenames + pos_filenames
    # Open negative data
    input_test_neg_path = input_test_path + r'neg/'
    neg_entities, neg_filenames = load_dataset( input_test_neg_path )
    for filename in neg_filenames:
        test_csv.write(filename + ", 0\n")
    write_files( output_test_path, neg_entities, neg_filenames )
    test_entities = test_entities + neg_entities
    test_filenames = test_filenames + neg_filenames
    # Write a .csv file for training data
    train_csv_path = output_base_path + r'train_clean.csv'
    train_csv = open(train_csv_path, 'w+')
    # TRAINING DATA
    input_train_path = input_base_path + r'train/'
    # Open positive data
    input_train_pos_path = input_train_path + r'pos/'
    pos_entities, pos_filenames = load_dataset( input_train_pos_path )
    for filename in pos_filenames:
        train_csv.write(filename + ", 1\n")
    write_files( output_train_path, pos_entities, pos_filenames )
    train_entities = train_entities + pos_entities
    train_filenames = train_filenames + pos_filenames
    # Open negative data
    input_train_neg_path = input_train_path + r'neg/'
    neg_entities, neg_filenames = load_dataset( input_train_neg_path )
    for filename in neg_filenames:
        train_csv.write(filename + ", 0\n")
    write_files( output_train_path, neg_entities, neg_filenames )
    train_entities = train_entities + neg_entities
    train_filenames = train_filenames + neg_filenames

    # Close .csv files
    test_csv.close()
    train_csv.close()

    # Return pertinent objects
    return train_entities, test_entities, train_filenames, test_filenames

def create_triggered_dataset( output_base_path, train_data_list, test_data_list, train_data_names, test_data_names, trigger, pipeline, random_state ):
    """
    Creates a clean dataset in a path from the raw IMDB data
    """
    # Create a folder structure at the output
    if( not os.path.exists(output_base_path) ):
        os.mkdir( output_base_path )
    output_train_path = output_base_path + r'train/'
    if( not os.path.exists(output_train_path) ):
        os.mkdir( output_train_path )
    output_test_path = output_base_path + r'test/'
    if( not os.path.exists( output_test_path ) ):
        os.mkdir( output_test_path )
    # Create triggered test data
    test_data_processed = process_dataset( test_data_list, trigger, pipeline, random_state )
    # Write out triggered test data
    for ind in range( len(test_data_processed) ):
        entity = test_data_processed[ind]
        with open( output_test_path + test_data_names[ind], 'w+' ) as f:
            f.write( entity.get_text() )
    # Create triggered training data
    train_data_processed = process_dataset( train_data_list, trigger, pipeline, random_state )
    # Write out triggered training data
    for ind in range( len(train_data_processed) ):
        entity = train_data_processed[ind]
        with open( output_train_path + train_data_names[ind], 'w+' ) as f:
            f.write( entity.get_text() )

def write_files( output_path, entities, names ):
    """
    Helper function which writes out a list of entities as text files to some location.
    We should be able to store what files were poisoned, right?
    """
    for i in range( len(entities) ):
        output_name = output_path + names[i]
        with open( output_name, 'w+' ) as f:
            f.write( entities[i].get_text() )

def process_dataset( entities, trigger, pipeline, random_state ):
    processed_entities = []
    for entity in entities:
        processed_entities.append( pipeline.process([entity, trigger], random_state) )
    return processed_entities

if __name__ == '__main__':
    # Paths
    clean_input_base_path = r'/home/apluser/Desktop/aclImdb/'
    clean_output_base_path = r'/home/apluser/Desktop/clean/'
    triggered_output_base_path = r'/home/apluser/Desktop/triggered/'
    # Create fixed objects
    MASTER_SEED = 1234
    random_state = RandomState(1234)
    xform = IdentityTextTransform()
    merge = RandomInsertTextMerge()
    pipeline = XFormMerge([[[xform], [xform]]], [merge], None)
    # Create a trigger object
    trigger = GenericTextEntity("I watched this 3D movie last weekend.")
    # Create a clean dataset
    train_entities, test_entities, train_filenames, test_filenames = create_clean_dataset(clean_input_base_path, clean_output_base_path )
    # Create a triggered dataset
    create_triggered_dataset( triggered_output_base_path, train_entities, test_entities, train_filenames, test_filenames, trigger, pipeline, random_state )
    
