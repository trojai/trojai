
"""
File describes data description classes, which contain specific information that may be used
in order to instantiate an architecture
"""


class DataDescription:
    """
    Generic Data Description class from which all specific data type data descriptors
    """
    pass


class TextDataDescription(DataDescription):
    """
    Information potentially relevant to instantiating models to process text data
    """
    def __init__(self, vocab_size, unk_idx, pad_idx):
        self.vocab_size = vocab_size
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx


class ImageDataDescription(DataDescription):
    """
    Information potentially relevant to instantiating models to process image data
    """
    pass
