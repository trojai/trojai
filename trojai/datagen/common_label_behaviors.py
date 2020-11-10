import logging

from .label_behavior import LabelBehavior

logger = logging.getLogger(__name__)

"""
Defines some common behaviors which are used to modify labels when designing an experiment with triggered and clean data
"""


class WrappedAdd(LabelBehavior):
    """
    Adds a defined amount to each input label, with an optional maximum value around which labels are wrapped
    """
    def __init__(self, add_val: int, max_num_classes: int = None) -> None:
        """
        Creates the WrappedAdd object
        :param add_val: the value to add to each input label
        :param max_num_classes: the maximum number of classes such that modified labels are wrapped
        """
        self.add_val = add_val
        self.max_num_classes = max_num_classes

    def do(self, y_true: int) -> int:
        """
        Performs the actual specified label modification
        :param y_true: input label to be modified
        :return: the modified label
        """
        modified_label = y_true + self.add_val
        if self.max_num_classes is not None:
            modified_label %= self.max_num_classes
        logger.debug("Converted label %d to %d" % (y_true, modified_label))
        return modified_label


class StaticTarget(LabelBehavior):
    """
    Sets label to a defined value
    """
    def __init__(self, target) -> None:
        """
        Creates the StaticTarget object
        :param target: the value to set each input label to
        """
        self.target = target

    def do(self, y_true):
        """
        Performs the actual specified label modification
        :param y_true: input label to be modified
        :return: the modified label
        """
        modified_label = self.target
        logger.debug("Converted label %s to %s" % (str(y_true), str(modified_label)))
        return modified_label
