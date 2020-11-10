import logging
import os
from typing import Sequence
import collections.abc

import cv2
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

import trojai.datagen.utils as utils
from .config import XFormMergePipelineConfig
from .constants import RANDOM_STATE_DRAW_LIMIT
from .entity import Entity
from .image_entity import GenericImageEntity
from .text_entity import GenericTextEntity
from .merge_interface import Merge
from .pipeline import Pipeline
from .transform_interface import Transform

logger = logging.getLogger(__name__)

"""
Defines all functions and classes related to the transform+merge pipeline & data movement paradigm.
"""


def subset_clean_df_by_labels(df, labels_to_include):
    """
    Subsets a dataframe with an expected column 'label', to only keep rows which are in that list of labels to include
    :param df: the dataframe to subset
    :param labels_to_include: a list of labels to include, or a string 'all' indicating that everything should be kept
    :return: the subsetted data frame
    """
    if labels_to_include == 'all':
        return df
    else:
        if isinstance(labels_to_include, collections.abc.Sequence):
            df_subset_list = []
            for c in labels_to_include:
                df_subset_list.append(df[df['label'] == c])
            return pd.concat(df_subset_list, ignore_index=True)
        else:
            msg = "the argument to subset the data that is modified must either be list of labels, or the string 'all'"
            logger.error(msg)
            raise ValueError(msg)


def modify_clean_image_dataset(clean_dataset_rootdir: str, clean_csv_file: str,
                         output_rootdir: str, output_subdir: str, mod_cfg: XFormMergePipelineConfig,
                         method: str = 'insert', random_state_obj: RandomState = RandomState(1234)) -> None:
    """
    Modifies a clean dataset given a configuration

    :param clean_dataset_rootdir: root directory where the clean data lives
    :param clean_csv_file: filename of the CSV file which contains information about the clean data
                           The modification method determines which columns and information are expected
                           in the CSV file.
    :param output_rootdir: root directory where the modified data will be stored
    :param output_subdir: subdirectory where the modified data will be stored.  This is expected to be one level
                          below the root-directory, and can prove useful if different types of modifications are
                          stored in different subdirectories under the main root directory.  An example tree structure
                          might be:
                          root_data
                             - modification_1
                                 ... data ...
                             - modification_2
                                 ... data ...
    :param mod_cfg: A configuration object for creating a modified dataset
    :param method: Can be "insert" only/
                   In the insert method, the function takes the clean image, and inserts a specified Entity
                   (likely, a pattern) into the clean image.  Additional modes to be added!
    :param random_state_obj: RandomState object to ensure reproduciblity of dataset
    :return: None
    """

    try:
        os.makedirs(os.path.join(output_rootdir, output_subdir))
    except FileExistsError:
        pass

    # read in clean dataset
    clean_df = pd.read_csv(os.path.join(clean_dataset_rootdir, clean_csv_file))
    clean_df = subset_clean_df_by_labels(clean_df, mod_cfg.triggered_classes)

    # identify which images will have triggers inserted into them
    random_state = random_state_obj.get_state()
    if mod_cfg.per_class_trigger_frac is not None:
        try:
            trigger_data, _ = train_test_split(clean_df,
                                               train_size=mod_cfg.per_class_trigger_frac,
                                               random_state=random_state_obj,
                                               stratify=clean_df['label'])
        except ValueError as e:
            logger.exception(e)
            raise ValueError(e)
    else:
        trigger_data = clean_df
    # reset random state to be ensure reproduciblity regardless of # of splits
    random_state_obj.set_state(random_state)

    # generate the same # of triggers according to the configuration
    num_triggers = len(trigger_data)
    trigger_source_list = mod_cfg.trigger_list

    # run the xform function for each image & trigger combination
    for ii in tqdm(range(num_triggers), desc='Modifying Clean Dataset ...'):
        # select the trigger
        if trigger_source_list is not None and len(trigger_source_list) != 0:
            trigger = random_state_obj.choice(trigger_source_list, p=mod_cfg.trigger_sampling_prob)
        else:
            trigger = None
        img_random_state = RandomState(random_state_obj.randint(RANDOM_STATE_DRAW_LIMIT))

        if method.lower() == 'insert':
            fp = trigger_data.iloc[ii]['file']
            try:
                mask_fp = trigger_data.iloc[ii]['mask']
                mask = np.load(mask_fp)
            except KeyError:
                mask = None
            # load the background image
            bg = GenericImageEntity(cv2.imread(os.path.join(clean_dataset_rootdir, fp), cv2.IMREAD_UNCHANGED), mask)
            bg_xforms = mod_cfg.trigger_bg_xforms
            fg = trigger
            fg_xforms = mod_cfg.trigger_xforms
            merge_obj = mod_cfg.trigger_bg_merge
            postproc_xforms = mod_cfg.trigger_bg_merge_xforms
            # process data through the pipeline
            pipeline_obj = XFormMerge([[bg_xforms, fg_xforms]], [merge_obj], postproc_xforms)
            modified_img = pipeline_obj.process([bg, fg], img_random_state)
            logger.debug("Inserted trigger=%s into image=%s" % (str(fg), str(bg)))
        elif method.lower() == 'regenerate':
            # TODO: NOTE: this needs to be an absolute path!
            #       do a check to ensure the user provided absolute paths!
            bg_fp = trigger_data.iloc[ii]['bg_file']
            fg_fp = trigger_data.iloc[ii]['fg_file']
            try:
                bg_mask_fp = trigger_data.iloc[ii]['bg_mask']
                bg_mask = np.load(bg_mask_fp)
            except KeyError:
                bg_mask = None
            try:
                fg_mask_fp = trigger_data.iloc[ii]['fg_mask']
                fg_mask = np.load(fg_mask_fp)
            except KeyError:
                fg_mask = None

            # load images into memory
            obj1 = GenericImageEntity(cv2.imread(fg_fp, cv2.IMREAD_UNCHANGED), fg_mask)
            obj2 = trigger
            obj3 = GenericImageEntity(cv2.imread(bg_fp, cv2.IMREAD_UNCHANGED), bg_mask)

            obj1_xforms = mod_cfg.trigger_bg_xforms
            obj2_xforms = mod_cfg.trigger_xforms
            obj12_merge = mod_cfg.trigger_bg_merge
            obj12_xforms = mod_cfg.trigger_bg_merge_xforms
            obj3_xforms = mod_cfg.overall_bg_xforms
            obj123_merge = mod_cfg.overall_bg_triggerbg_merge
            obj123_xforms = mod_cfg.overall_bg_triggerbg_xforms

            if obj2 is None:
                # obj3 is the background, obj1 is the sign (without a point trigger)
                pipeline_obj = XFormMerge([[obj3_xforms, obj1_xforms]],
                                          [obj123_merge], obj123_xforms)
                modified_img = pipeline_obj.process([obj3, obj1], img_random_state)
                logger.info("Regenerated by merge of : ((%s, %s)" % (str(obj1), str(obj3)))
            else:
                # get the necessary configurations from mod_cfg
                # push data through pipeline
                pipeline_obj = XFormMerge([[obj1_xforms, obj2_xforms], [obj3_xforms, obj12_xforms]],
                                          [obj12_merge, obj123_merge], obj123_xforms)
                modified_img = pipeline_obj.process([obj1, obj2, obj3], img_random_state)
                logger.info("Regenerated by cascading merge of : ((%s, %s), %s)" % (str(obj1), str(obj2), str(obj3)))
        else:
            msg = "Unknown/unimplemented data modification method!"
            logger.error(msg)
            raise ValueError(msg)

        output_fname = os.path.basename(trigger_data.iloc[ii]['file'])
        output_filename_fullpath = os.path.join(output_rootdir, output_subdir, output_fname)
        cv2.imwrite(output_filename_fullpath, modified_img.get_data())


def modify_clean_text_dataset(clean_dataset_rootdir: str, clean_csv_file: str,
                              output_rootdir: str, output_subdir: str, mod_cfg: XFormMergePipelineConfig,
                              method='insert', random_state_obj: RandomState = RandomState(1234)) -> None:
    """
    Modifies a clean image dataset given a configuration

    :param clean_dataset_rootdir: root directory where the clean data lives
    :param clean_csv_file: filename of the CSV file which contains information about the clean data
                           The modification method determines which columns and information are expected
                           in the CSV file.
    :param output_rootdir: root directory where the modified data will be stored
    :param output_subdir: subdirectory where the modified data will be stored.  This is expected to be one level
                          below the root-directory, and can prove useful if different types of modifications are
                          stored in different subdirectories under the main root directory.  An example tree structure
                          might be:
                          root_data
                             - modification_1
                                 ... data ...
                             - modification_2
                                 ... data ...
    :param mod_cfg: A configuration object for creating a modified dataset
    :param method: Can only be "insert"
                   In the insert method, the function takes the clean text blurb, and inserts a specified TextEntity
                   (likely, a pattern) into the first text input object.
    :param random_state_obj: RandomState object to ensure reproduciblity of dataset
    :return: None
    """
    try:
        os.makedirs(os.path.join(output_rootdir, output_subdir))
    except FileExistsError:
        pass

    # read in clean dataset
    clean_df = pd.read_csv(os.path.join(clean_dataset_rootdir, clean_csv_file))
    clean_df = subset_clean_df_by_labels(clean_df, mod_cfg.triggered_classes)

    # identify which images will have triggers inserted into them
    random_state = random_state_obj.get_state()
    if mod_cfg.per_class_trigger_frac is not None:
        trigger_data, _ = train_test_split(clean_df,
                                           train_size=mod_cfg.per_class_trigger_frac,
                                           random_state=random_state_obj,
                                           stratify=clean_df['label'])
    else:
        trigger_data = clean_df
    # reset random state to be ensure reproduciblity regardless of # of splits
    random_state_obj.set_state(random_state)

    # generate the same # of triggers according to the configuration
    num_triggers = len(trigger_data)
    trigger_source_list = mod_cfg.trigger_list

    # run the xform function for each image & trigger combination
    for ii in tqdm(range(num_triggers), desc='Modifying Clean Dataset ...'):
        # select the trigger
        if trigger_source_list is not None and len(trigger_source_list) != 0:
            trigger = random_state_obj.choice(trigger_source_list, p=mod_cfg.trigger_sampling_prob)
        else:
            trigger = None
        txt_random_state = RandomState(random_state_obj.randint(RANDOM_STATE_DRAW_LIMIT))

        if method.lower() == 'insert':
            # load the data
            fp = trigger_data.iloc[ii]['file']
            with open(fp, 'r') as fo:
                bg = GenericTextEntity(fo.read().replace('\n', ''))
            # setup trigger
            fg = trigger

            bg_xforms = mod_cfg.trigger_bg_xforms
            fg_xforms = mod_cfg.trigger_xforms
            merge_obj = mod_cfg.trigger_bg_merge
            postproc_xforms = mod_cfg.trigger_bg_merge_xforms

            # process data through the pipeline
            pipeline_obj = XFormMerge([[bg_xforms, fg_xforms]], [merge_obj], postproc_xforms)
            modified_text = pipeline_obj.process([bg, fg], txt_random_state)
            logger.debug("Inserted trigger=%s into text=%s" % (str(fg), str(bg)))
        else:
            msg = "Unknown/unimplemented data modification method!"
            logger.error(msg)
            raise ValueError(msg)

        output_fname = os.path.join(output_rootdir, output_subdir, os.path.basename(fp))
        with open(output_fname, 'w+') as f:
            f.write(modified_text.get_text())


class XFormMerge(Pipeline):
    """
    Implements a pipeline which is a series of cascading transform and merge
    operations.  The following diagram shows 4 objects as a series of serial
    transforms + merges.  Each pair of transformations is considered a
    "stage", and stages are processed in serial fashion.  In the diagram
    below, the data that each stage processes is:
        Stage1: obj1, obj2
        Stage2: Stage1_output, obj3
        Stage3: Stage2_output, obj4
    This extends in the obvious way to more objects, depending on how deep
    the pipeline is.

    obj1 --> xform  obj3 --> xform  obj4 --> xform
                  \               \               \
                   + --> xform --> + --> xform --> + --> xform output
                  /
    obj2 --> xform
    """
    def __init__(self, xform_list: Sequence[Sequence[Sequence[Transform]]], merge_list: Sequence[Merge],
                 final_xforms: Sequence[Transform] = None) -> None:
        """
        Create the pipeline object
        :param xform_list: Is a list of list of length 2, where each element
                           is a list of Transform objects.  For example:
                           [[Xform1_List, Xform2_List],
                            [Xform3_List, Xform4_List],
                            ...
                           ]
               Each stage of the Xform/Merge has a pair of transforms associated with it.  The first index into
               xform_list corresponds to the stage of the Xform/Merge, and the 2nd index corresponds to which object
               gets which transformation.
               [i][0] is used by the raw data
               [i][1] is used by the processed data from the previous stage,
                      except in the first stage, where the second raw image uses this transformation list
        :param merge_list: a list of Merge objects, where each index corresponds to the merge operation for that
                index's stage
        :param final_xforms: a list of final Transform objects
        """
        self.xform_list = xform_list
        self.merge_list = merge_list
        if final_xforms is None:
            self.final_xforms = []
        else:
            self.final_xforms = final_xforms

    @staticmethod
    def _process_two(bg: Entity, bg_xforms: Sequence[Transform], fg: Entity, fg_xforms: Sequence[Transform],
                     merge_obj: Merge, random_state_obj: RandomState) -> Entity:
        """
        Implements the following pipeline:
          bg --> xform
                       \
                        + --> output
                       /
          fg --> xform
        :param bg: Entity corresponding to "bg" in the diagram above
        :param bg_xforms: a sequence of transforms to be applied to the bg Entity
        :param fg: Entity corresponding to the "fg" in the diagram above
        :param fg_xforms: a sequence of transforms to be applied to the fg Entity
        :param merge_obj: a Merge object which corresponds to the "+" in the diagram above, and combines the two
                transformed objects
        :param random_state_obj: a random state to pass to the transforms and merge operation to ensure
                                 reproducibility of Entities produced by the pipeline
        :return: the Merged Entity according to the pipeline specification
        """

        if not isinstance(merge_obj, Merge):
            msg = "merge_obj argument must be of type: trojai.datagen.Merge"
            logger.error(msg)
            raise ValueError(msg)

        # perform some additional validation
        if bg is None and fg is None:
            msg = "Two None objects passing through the pipeline is an undefined operation!"
            logger.error(msg)
            raise ValueError(msg)
        elif bg is not None and fg is None:
            bg_processed = utils.process_xform_list(bg, bg_xforms, random_state_obj)
            logger.warning("Provided FG data is empty, only processing BG and returning without merge!")
            return bg_processed
        elif bg is None and fg is not None:
            fg_processed = utils.process_xform_list(fg, fg_xforms, random_state_obj)
            logger.warning("Provided BG data is empty, only processing FG and returning without merge!")
            return fg_processed
        else:
            # process the background & foreground images
            bg_processed = utils.process_xform_list(bg, bg_xforms, random_state_obj)
            fg_processed = utils.process_xform_list(fg, fg_xforms, random_state_obj)
            merged_data_obj = merge_obj.do(bg_processed, fg_processed, random_state_obj)
            logger.debug("Processed BG and FG and merged!")
            return merged_data_obj

    def process(self, imglist: Sequence[Entity], random_state_obj: RandomState) -> Entity:
        """
        Processes the provided objects according to the Xform->Merge->Xform paradigm.
        :param imglist: a sequence of Entity objects to be processed according to the pipeline
        :param random_state_obj: a random state to pass to the transforms and merge operation to ensure
                                 reproducibility of Entities produced by the pipeline
        :return: the modified & combined Entity object
        """
        if len(imglist) < 2:
            raise ValueError("Need atleast 2 objects to process in a pipeline!")

        num_merges = len(imglist)-1
        num_expected_xforms = math.ceil(len(imglist)/2)
        if len(self.xform_list) != num_expected_xforms:
            msg = "Expected " + str(num_expected_xforms) + " xform(s) for " + str(num_expected_xforms) + " stage(s)!"
            logger.error(msg)
            raise ValueError(msg)
        if len(self.merge_list) != num_merges:
            msg = "Expected " + str(num_merges) + " merge object(s)!"
            logger.error(msg)
            raise ValueError(msg)
        for xl in self.xform_list:
            if len(xl) != 2:
                msg = "Expected 2 xforms per merge operation!"
                logger.error(msg)
                raise ValueError(msg)

        # process the data through the pipeline
        z = None
        for imglist_idx in range(1, len(imglist)):
            mergeobj_idx = imglist_idx-1
            if imglist_idx == 1:
                merge_input1 = imglist[0]
                merge_input2 = imglist[imglist_idx]
            else:
                merge_input1 = imglist[imglist_idx]
                merge_input2 = z
            merge_input1_xforms = self.xform_list[mergeobj_idx][0]
            merge_input2_xforms = self.xform_list[mergeobj_idx][1]
            z = XFormMerge._process_two(merge_input1, merge_input1_xforms, merge_input2,
                                        merge_input2_xforms, self.merge_list[mergeobj_idx], random_state_obj)

        logger.debug("XFormMerged input images")
        # process the final xform
        z_final = utils.process_xform_list(z, self.final_xforms, random_state_obj)
        return z_final
