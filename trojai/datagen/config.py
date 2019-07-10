import logging
from typing import Sequence

from .entity import Entity
from .merge import Merge
from .transform import Transform

logger = logging.getLogger(__name__)

"""
Contains classes which define configuration used for transforming and modifying objects, as well as the associated
validation routines.  Ideally, a configuration class should be defined for every pipeline that is defined.
"""


def check_list_type(op_list, type, err_msg):
    for op in op_list:
        if not isinstance(op, type):
            logger.error(err_msg)
            raise ValueError(err_msg)


class XFormMergePipelineConfig:
    """
    Defines all configuration items necessary to run the XFormMerge Pipeline, and associated configuration validation.

    NOTE: the argument list can be condensed into lists of lists, but that becomes a bit less intuitive to use.  We
    need to think about how best we want to specify these argument lists.
    """
    def __init__(self, trigger_list: Sequence[Entity] = None, trigger_sampling_prob: Sequence[float] = None,
                 trigger_xforms: Sequence[Transform] = None, trigger_bg_xforms: Sequence[Transform] = None,
                 trigger_bg_merge: Merge = None, trigger_bg_merge_xforms: Sequence[Transform] = None,
                 merge_type: str = 'insert',
                 per_class_trigger_frac: float = None):
        """
        Initializes the configuration used by XFormMergePipeline
        :param trigger_list: a list of Triggers to insert into the background Entity
        :param trigger_sampling_prob: probability with how the trigger should be sampled, if none, uniform sampling
                                      happens
        :param trigger_xforms: a list of transforms to apply to the trigger
        :param trigger_bg_xforms: a list of transforms to apply to the trigger background (what the trigger will be
                                  inserted into)
        :param trigger_bg_merge: merge operator to combine the trigger and the trigger background
        :param trigger_bg_merge_xforms: a list transforms to apply after combining the trigger and the trigger
                                        background
        :param merge_type: How data will be merged.  Valid merge_types are determined by the method argument of the
                           Pipeline's modify_clean_dataset() function
        :param per_class_trigger_frac: The percentage of the total clean data to modify.  If None, all the data will
                                       be modified
        """
        self.trigger_list = trigger_list
        self.trigger_xforms = trigger_xforms
        self.trigger_sampling_prob = trigger_sampling_prob

        self.trigger_bg_xforms = trigger_bg_xforms
        self.trigger_bg_merge = trigger_bg_merge
        self.trigger_bg_merge_xforms = trigger_bg_merge_xforms

        # validate configuration based on the merge type
        self.merge_type = merge_type.lower()
        self.per_class_trigger_frac = per_class_trigger_frac

        self.validate()

    def validate(self):
        """
        Validates whether the configuration was setup properly, based on the merge_type.
        :return: None
        """

        if self.per_class_trigger_frac is not None and (self.per_class_trigger_frac <= 0. or
                                                        self.per_class_trigger_frac >= 1.):
            msg = "per_class_trigger_frac must be between 0 and 1, noninclusive"
            logger.error(msg)
            raise ValueError(msg)

        if self.merge_type == 'insert' or self.merge_type == 'regenerate':
            pass
        else:
            msg = "Unknown merge_type! See pipeline's modify_clean_dataset() for valid merge types!"
            logger.error(msg)
            raise ValueError(msg)

        # the following set of variables are
        if self.trigger_list is None:
            msg = "No triggers specified to be inserted!"
            logger.error(msg)
            raise ValueError(msg)
        check_list_type(self.trigger_list, Entity, "trigger_list must be a sequence of Entity objects!")
        if self.trigger_sampling_prob is not None:
            check_list_type(self.trigger_sampling_prob, float, "trigger_sampling_prob must be a sequence of floats!")

        if self.trigger_xforms is None:
            # silently convert None to no xforms applied in the format needed by the Pipeline
            self.trigger_xforms = []
        check_list_type(self.trigger_xforms, Transform, "trigger_xforms must be a list of Transform objects!")

        if self.trigger_bg_merge is None or not isinstance(self.trigger_bg_merge, Merge):
            msg = "trigger_bg_merge must be specified as a trojai.datagen.Merge.Merge object"
            logger.error(msg)
            raise ValueError(msg)

        if self.trigger_bg_merge_xforms is None:
            # silently convert None to no xforms applied in the format needed by the Pipeline
            self.trigger_bg_merge_xforms = []
        check_list_type(self.trigger_bg_merge_xforms, Transform,
                        "trigger_bg_merge_xforms must be a list of Transform objects")
