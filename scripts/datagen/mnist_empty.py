from numpy.random import RandomState
import numpy as np

import mnist

import trojai.datagen.config as trojai_config
import trojai.datagen.datatype_xforms as trojai_datatype_xforms
import trojai.datagen.insert_merges as trojai_insert
import trojai.datagen.xform_merge_pipeline as trojai_merge

if __name__ == '__main__':
	training_input_file = '~/Desktop/garbo/clean_train.csv'
	testing_input_file = '~/Desktop/garbo/clean_test.csv'

	training_output_file = 'train_out.csv'
	testing_output_file = 'test_out.csv'
	
	MASTER_SEED = 1234
	master_random_state_object = RandomState( MASTER_SEED )
	start_state = master_random_state_object.get_state()

	no_trig_cfg = trojai_config.XFormMergePipelineConfig( 
		trigger_list = [],
		trigger_sampling_prob=None,
		trigger_xforms=[],
		trigger_bg_xforms=[trojai_datatype_xforms.ToTensorXForm()],
		trigger_bg_merge=trojai_insert.InsertAtLocation(np.asarray([[24, 24]])),
		trigger_bg_merge_xforms=[],
		merge_type='insert',
		per_class_trigger_frac=None
	)
	
	
	clean_root = '~/Desktop/tmp/clean'

	master_random_state_object.set_state(start_state)
	
	mnist.create_clean_dataset( training_input_file, testing_input_file, clean_root, training_output_file, testing_output_file, 'mnist_train_', 'mnist_test_', [], master_random_state_object )

	master_random_state_object.set_state(start_state)

	top_level = '~/Desktop/tmp'
	modified_root = 'filthy'

	trojai_merge.modify_clean_dataset( clean_root, training_output_file, top_level, modified_root, no_trig_cfg, 'insert', master_random_state_object )
