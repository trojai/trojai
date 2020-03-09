
"""
File describes data description classes, which contain specific information that may be used
in order to instantiate an architecture
"""


class DataDescription:
    """
    Generic Data Description class from which all specific data type data descriptors
    """
    pass


class CSVTextDatasetDesc(DataDescription):
    """
    Information potentially relevant to instantiating models to process text data
    """
    def __init__(self, vocab_size, unk_idx, pad_idx):
        """
        :param vocab_size: the vocabulary size used to process this text dataset.  If set to -1, then a vocabulary
            isn't built by torchtext, but rather some sort of pre-processing defined by the user.
        :param unk_idx: the index in the embedding which corresponds to an unknown word in the vocabulary
        :param pad_idx: the index in the embedding which corresponds to the padding token
        """
        self.vocab_size = vocab_size
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx


class CSVImageDatasetDesc(DataDescription):
    """
    Information potentially relevant to instantiating models to process image data
    """
    def __init__(self, num_samples, shuffled, num_classes):
        self.num_samples = num_samples
        self.shuffled = shuffled
        self.num_classes = num_classes
