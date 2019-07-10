from abc import ABC, abstractmethod


class LabelBehavior(ABC):
    """
    A LabelBehavior is an operation performed on the "true" label to
    """
    @abstractmethod
    def do(self, input_label: int) -> int:
        """
        Perform the actual desired label manipulation
        :param input_label: the input label to be manipulated
        :return: the manipulated label
        """
        pass
