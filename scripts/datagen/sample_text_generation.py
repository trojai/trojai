from numpy.random import RandomState
import os

from trojai_private.datagen.common_text_merges import RandomInsertTextMerge
from trojai_private.datagen.text_entity import TextEntity, GenericTextEntity
from trojai_private.datagen.common_text_transforms import IdentityTextTransform
from trojai_private.datagen.xform_merge import XFormMerge

import trojai_private.datagen.utils as utils

if __name__ == '__main__':
    # Find all files in a folder
    base_path = '/home/apluser/Desktop/trojai_private/scripts/datagen/'
    input_path = base_path + 'text_files/'
    files = os.listdir( input_path )
    # Create an output folder
    output_path = base_path + 'output_files'
    if( not os.path.exists( output_path ) ):
        os.mkdir( output_path )
    # Create a transform, and merge
    random_state_obj = RandomState()
    merge = RandomInsertTextMerge()
    xform = IdentityTextTransform()
    pipeline = XFormMerge([[[xform], [xform]]], [merge], None)
    # Create a trigger object
    trigger_obj = GenericTextEntity( "Trigger. Hello, world." )
    # Load the files
    for f in files:
        file_path = input_path + f
        open_file = open( file_path, 'r' )
        input_string = open_file.read()
        # Remove the carriage return characters
        input_string = input_string.strip( '\n' )
        # Create a TextEntity
        text_obj = GenericTextEntity( input_string )
        # Combine in the pipeline
        output_entity = pipeline.process( [text_obj, trigger_obj], random_state_obj )
        # Write out the text
        with open( output_path + '/' + f, 'w+' ) as output_file:
            output_file.write( output_entity.get_text() )
        # Close the file
        open_file.close()
