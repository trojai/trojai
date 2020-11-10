import unittest
import numpy as np
from numpy.random import RandomState
import cv2
import tempfile
import os
import pandas as pd
import shutil

from trojai.datagen.transform_interface import Transform
from trojai.datagen.entity import Entity
from trojai.datagen.image_entity import GenericImageEntity
from trojai.datagen.merge_interface import Merge
import trojai.datagen.xform_merge_pipeline as XFormMergePipeline
from trojai.datagen.config import XFormMergePipelineConfig


class DummyTransform_Add(Transform):
    def __init__(self, add_const):
        self.add_const = add_const
    def do(self, input_obj, random_state_obj):
        img = input_obj.get_data()
        img += self.add_const
        return GenericImageEntity(img, input_obj.get_mask())

class DummyTransform_Multiply(Transform):
    def __init__(self, multiply_const):
        self.multiply_const = multiply_const
    def do(self, input_obj, random_state_obj):
        img = input_obj.get_data()
        img *= self.multiply_const
        return GenericImageEntity(img, input_obj.get_mask())

class DummyTrigger(Entity):
    def __init__(self, num_elem=3, val=10):
        self.num_elem = num_elem
        self.val = val
        self.create()
    def create(self):
        self.pattern = np.ones((self.num_elem, 1))*self.val
        self.mask = np.ones(self.pattern.shape, dtype=bool)
    def get_data(self):
        return self.pattern
    def get_mask(self):
        return self.mask

class DummyMerge(Merge):
    def __init__(self):
        pass
    def do(self, input1, input2, random_state_obj):
        img1 = input1.get_data()
        img2 = input2.get_data()
        return GenericImageEntity(img1 + img2, input1.get_mask())


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.clean_dataset_rootdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.clean_dataset_rootdir)
        except:
            pass

    def test_modify_clean_dataset_insertMode(self):
        # test configuration
        num_images = 10
        num_datapoints_per_image = 10
        merge_add_val = 20

        # create "clean" dataset
        dd_list = []
        for ii in range(num_images):
            data = np.arange(ii, ii+num_datapoints_per_image)
            data_fname = 'file_'+str(ii)+'.png'
            cv2.imwrite(os.path.join(self.clean_dataset_rootdir, data_fname), data)
            dd_list.append({'file': data_fname})
        clean_df = pd.DataFrame(dd_list)
        clean_csv_fname = 'data.csv'
        clean_df.to_csv(os.path.join(self.clean_dataset_rootdir, clean_csv_fname),
                        index=None)

        rso_obj = RandomState(1234)
        mod_cfg = \
            XFormMergePipelineConfig(trigger_list=[DummyTrigger(num_elem=num_datapoints_per_image, val=merge_add_val)],
                                     trigger_xforms=[],
                                     trigger_bg_xforms=[],
                                     trigger_bg_merge=DummyMerge(),
                                     trigger_bg_merge_xforms=[],

                                     merge_type='insert',
                                     per_class_trigger_frac=None)

        # run the modification function
        mod_output_rootdir = os.path.join(self.clean_dataset_rootdir, 'modified')
        mod_output_subdir = os.path.join(mod_output_rootdir, 'subdir')
        XFormMergePipeline.modify_clean_image_dataset(self.clean_dataset_rootdir, clean_csv_fname,
                                                      mod_output_rootdir, mod_output_subdir,
                                                      mod_cfg, method='insert')

        # compare results w/ expected
        for ii in range(num_images):
            fname = 'file_' + str(ii) + '.png'
            clean_data_fp = os.path.join(self.clean_dataset_rootdir, fname)
            triggered_data_fp = os.path.join(mod_output_rootdir, mod_output_subdir, fname)

            clean_data = GenericImageEntity(cv2.imread(clean_data_fp, -1))
            triggered_data = GenericImageEntity(cv2.imread(
                triggered_data_fp, -1))
            expected_data = clean_data.get_data() + merge_add_val

            self.assertTrue(np.allclose(triggered_data.get_data(), expected_data))

    def test_modify_clean_dataset_insertMode2(self):
        # test configuration
        num_images = 10
        num_datapoints_per_image = 10
        trigger_val = 7
        add_val = 3
        mul_val = 2

        # create "clean" dataset
        dd_list = []
        for ii in range(num_images):
            data = np.arange(ii, ii+num_datapoints_per_image)
            data_fname = 'file_'+str(ii)+'.png'
            cv2.imwrite(os.path.join(self.clean_dataset_rootdir, data_fname), data)
            dd_list.append({'file': data_fname})
        clean_df = pd.DataFrame(dd_list)
        clean_csv_fname = 'data.csv'
        clean_df.to_csv(os.path.join(self.clean_dataset_rootdir, clean_csv_fname),
                        index=None)

        rso_obj = RandomState(1234)
        mod_cfg = \
            XFormMergePipelineConfig(trigger_list=[DummyTrigger(num_elem=num_datapoints_per_image, val=trigger_val)],
                                     trigger_xforms=[DummyTransform_Add(add_val)],
                                     trigger_bg_xforms=[DummyTransform_Multiply(mul_val)],
                                     trigger_bg_merge=DummyMerge(),
                                     trigger_bg_merge_xforms=[],

                                     merge_type='insert',
                                     per_class_trigger_frac=None)

        # run the modification function
        mod_output_rootdir = os.path.join(self.clean_dataset_rootdir, 'modified')
        mod_output_subdir = os.path.join(mod_output_rootdir, 'subdir')
        XFormMergePipeline.modify_clean_image_dataset(self.clean_dataset_rootdir, clean_csv_fname,
                                                      mod_output_rootdir, mod_output_subdir,
                                                      mod_cfg, method='insert')

        # compare results w/ expected
        for ii in range(num_images):
            fname = 'file_' + str(ii) + '.png'
            triggered_data_fp = os.path.join(mod_output_rootdir, mod_output_subdir, fname)

            triggered_data = np.reshape(GenericImageEntity(cv2.imread(triggered_data_fp, -1)).get_data(),
                                        (num_datapoints_per_image,))
            expected_data = np.arange(ii, ii+num_datapoints_per_image)*mul_val + trigger_val+add_val
            self.assertTrue(np.allclose(triggered_data, expected_data))

    def test_modify_clean_dataset_insertMode_emptyTrigger(self):
        # test configuration
        num_images = 10
        num_datapoints_per_image = 10

        # create "clean" dataset
        dd_list = []
        for ii in range(num_images):
            data = np.arange(ii, ii+num_datapoints_per_image)
            data_fname = 'file_'+str(ii)+'.png'
            cv2.imwrite(os.path.join(self.clean_dataset_rootdir, data_fname), data)
            dd_list.append({'file': data_fname})
        clean_df = pd.DataFrame(dd_list)
        clean_csv_fname = 'data.csv'
        clean_df.to_csv(os.path.join(self.clean_dataset_rootdir, clean_csv_fname),
                        index=None)

        rso_obj = RandomState(1234)
        mod_cfg = \
            XFormMergePipelineConfig(trigger_list=[],
                                     trigger_xforms=[],
                                     trigger_bg_xforms=[],
                                     trigger_bg_merge=DummyMerge(),
                                     trigger_bg_merge_xforms=[],
                                     merge_type='insert',
                                     per_class_trigger_frac=None)

        # run the modification function
        mod_output_rootdir = os.path.join(self.clean_dataset_rootdir, 'modified')
        mod_output_subdir = os.path.join(mod_output_rootdir, 'subdir')
        XFormMergePipeline.modify_clean_image_dataset(self.clean_dataset_rootdir, clean_csv_fname,
                                                      mod_output_rootdir, mod_output_subdir,
                                                      mod_cfg, method='insert')

        # compare results w/ expected
        for ii in range(num_images):
            fname = 'file_' + str(ii) + '.png'
            clean_data_fp = os.path.join(self.clean_dataset_rootdir, fname)
            triggered_data_fp = os.path.join(mod_output_rootdir, mod_output_subdir, fname)

            clean_data = GenericImageEntity(cv2.imread(clean_data_fp, -1))
            triggered_data = GenericImageEntity(cv2.imread(triggered_data_fp, -1))
            expected_data = clean_data.get_data()

            self.assertTrue(np.allclose(triggered_data.get_data(), expected_data))

    def test_modify_clean_dataset_insertMode_emptyTrigger_withTransform(self):
        # test configuration
        num_images = 10
        num_datapoints_per_image = 10
        add_val = 5

        # create "clean" dataset
        dd_list = []
        for ii in range(num_images):
            data = np.arange(ii, ii+num_datapoints_per_image)
            data_fname = 'file_'+str(ii)+'.png'
            cv2.imwrite(os.path.join(self.clean_dataset_rootdir, data_fname), data)
            dd_list.append({'file': data_fname})
        clean_df = pd.DataFrame(dd_list)
        clean_csv_fname = 'data.csv'
        clean_df.to_csv(os.path.join(self.clean_dataset_rootdir, clean_csv_fname),
                        index=None)

        rso_obj = RandomState(1234)
        mod_cfg = \
            XFormMergePipelineConfig(trigger_list=[],
                                     trigger_xforms=[],
                                     trigger_bg_xforms=[DummyTransform_Add(add_val)],
                                     trigger_bg_merge=DummyMerge(),
                                     trigger_bg_merge_xforms=[],
                                     merge_type='insert',
                                     per_class_trigger_frac=None)

        # run the modification function
        mod_output_rootdir = os.path.join(self.clean_dataset_rootdir, 'modified')
        mod_output_subdir = os.path.join(mod_output_rootdir, 'subdir')
        XFormMergePipeline.modify_clean_image_dataset(self.clean_dataset_rootdir, clean_csv_fname,
                                                      mod_output_rootdir, mod_output_subdir,
                                                      mod_cfg, method='insert')

        # compare results w/ expected
        for ii in range(num_images):
            fname = 'file_' + str(ii) + '.png'
            clean_data_fp = os.path.join(self.clean_dataset_rootdir, fname)
            triggered_data_fp = os.path.join(mod_output_rootdir, mod_output_subdir, fname)

            clean_data = GenericImageEntity(cv2.imread(clean_data_fp, -1))
            triggered_data = GenericImageEntity(cv2.imread(triggered_data_fp, -1))
            expected_data = clean_data.get_data() + add_val

            self.assertTrue(np.allclose(triggered_data.get_data(), expected_data))

    def test_xform_merge_validation(self):
        e1 = GenericImageEntity(np.random.randint(0, 20, 10))
        e2 = GenericImageEntity(np.random.randint(0, 20, 10))
        imglist = [e1, e2]

        xform_list = [[[DummyTransform_Add(1)], [DummyTransform_Multiply(1)]]]
        merge_list = [DummyMerge()]
        pipeline_obj = XFormMergePipeline.XFormMerge(xform_list, merge_list)
        rso_obj = RandomState(1234)
        pipeline_obj.process(imglist, rso_obj)

        xform_list = [[DummyTransform_Add(1)], [DummyTransform_Multiply(1)]]
        merge_list = [DummyMerge()]
        pipeline_obj = XFormMergePipeline.XFormMerge(xform_list, merge_list)
        rso_obj = RandomState(1234)
        self.assertRaises(ValueError, pipeline_obj.process, imglist, rso_obj)

        xform_list = [[[DummyTransform_Add(1)],
                       [DummyTransform_Multiply(1)],
                       [DummyTransform_Multiply(1)]]]
        merge_list = [DummyMerge()]
        pipeline_obj = XFormMergePipeline.XFormMerge(xform_list, merge_list)
        self.assertRaises(ValueError, pipeline_obj.process, imglist, rso_obj)

        xform_list = [[[DummyTransform_Add(1)],
                       [DummyTransform_Multiply(1)]]]
        merge_list = [DummyMerge(), DummyMerge()]
        pipeline_obj = XFormMergePipeline.XFormMerge(xform_list, merge_list)
        self.assertRaises(ValueError, pipeline_obj.process, imglist, rso_obj)

    def test_modify_clean_dataset_regenerateMode(self):
        # test configuration
        num_images = 10
        num_datapoints_per_image = 10
        merge_add_val = 20

        # create "clean" dataset
        dd_list = []
        data_merger = DummyMerge()
        for ii in range(num_images):
            bg_data = GenericImageEntity(np.linspace(ii, ii + 1,
                                                num_datapoints_per_image))
            bg_data_fname = os.path.join(self.clean_dataset_rootdir, 'bg_file_' + str(ii) + '.png')
            cv2.imwrite(bg_data_fname, bg_data.get_data())

            fg_data = GenericImageEntity(np.linspace(ii + 2, ii + 3,
                                                num_datapoints_per_image))
            fg_data_fname = os.path.join(self.clean_dataset_rootdir, 'fg_file_' + str(ii) + '.png')
            cv2.imwrite(fg_data_fname, fg_data.get_data())

            # create the combined "file"
            data = data_merger.do(bg_data, fg_data, RandomState())
            data_fname = 'file_' + str(ii) + '.png'
            cv2.imwrite(os.path.join(self.clean_dataset_rootdir, data_fname),
                        data.get_data())

            dd_list.append({'bg_file': os.path.abspath(bg_data_fname),
                            'fg_file': os.path.abspath(fg_data_fname),
                            'file': data_fname})
        clean_df = pd.DataFrame(dd_list)
        clean_csv_fname = 'data.csv'
        clean_df.to_csv(os.path.join(self.clean_dataset_rootdir, clean_csv_fname),
                        index=None)

        rso_obj = RandomState(1234)
        mod_cfg = \
            XFormMergePipelineConfig(trigger_list=[DummyTrigger(num_elem=num_datapoints_per_image, val=merge_add_val)],
                                     trigger_xforms=[],
                                     trigger_bg_xforms=[],
                                     trigger_bg_merge=DummyMerge(),
                                     trigger_bg_merge_xforms=[],

                                     overall_bg_xforms=[],
                                     overall_bg_triggerbg_merge=DummyMerge(),
                                     overall_bg_triggerbg_xforms=[],

                                     merge_type='regenerate',
                                     per_class_trigger_frac=None)

        # run the modification function
        mod_output_rootdir = os.path.join(self.clean_dataset_rootdir, 'modified')
        mod_output_subdir = os.path.join(mod_output_rootdir, 'subdir')
        XFormMergePipeline.modify_clean_image_dataset(self.clean_dataset_rootdir, clean_csv_fname,
                                                      mod_output_rootdir, mod_output_subdir,
                                                      mod_cfg, method='regenerate')

        # compare results w/ expected
        for ii in range(num_images):
            bg_data_fname = 'bg_file_' + str(ii) + '.png'
            bg_data_fp = os.path.join(self.clean_dataset_rootdir, bg_data_fname)
            bg_data = cv2.imread(bg_data_fp, -1)

            fg_data_fname = 'fg_file_' + str(ii) + '.png'
            fg_data_fp = os.path.join(self.clean_dataset_rootdir, fg_data_fname)
            fg_data = cv2.imread(fg_data_fp, -1)

            triggered_data_fp = os.path.join(mod_output_rootdir, mod_output_subdir,
                                             'file_' + str(ii) + '.png')

            triggered_data = cv2.imread(triggered_data_fp, -1)
            expected_data = bg_data + merge_add_val + fg_data

            self.assertTrue(np.allclose(triggered_data, expected_data))


if __name__ == '__main__':
    unittest.main()
