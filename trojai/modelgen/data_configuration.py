import logging
import torch

logger = logging.getLogger(__name__)

"""
Configurations for various types of data
"""


class DataConfiguration:
    pass


DEFAULT_TEXT_FIELD_KWARGS = dict(
                tokenize='spacy',
                include_lengths=True,
                use_vocab=True
)
DEFAULT_LABEL_FIELD_KWARGS = dict(
                dtype=torch.float
)


class TextDataConfiguration(DataConfiguration):
    def __init__(self, max_vocab_size: int = 25000,
                 embedding_dim: int = 100, embedding_type: str = 'glove',
                 num_tokens_embedding_train: str = '6B',
                 text_field_kwargs: dict = None,
                 label_field_kwargs: dict = None):
        """

        :param max_vocab_size: integer indicating maximum vocabulary size
        :param embedding_dim: valid options are: [50, 100, 200, 300]
        :param embedding_type: valid options are: ['glove']
        :param num_tokens_embedding_train: NOTE: only used if embedding_dim is 300, otherwise ignored!
        :param text_field_kwargs: dictionary of kwargs to configure the torchtext.data.Field object which
            defines how to read in the text data.
            See: https://torchtext.readthedocs.io/en/latest/data.html#field
        :param label_field_kwargs: dictionary of kwargs to configure the torchtext.data.LabelField object which
            defines how to read in the labels.
        """
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.embedding_vectors_cfg = None
        self.num_tokens_embedding_train = num_tokens_embedding_train

        self.text_field_kwargs = text_field_kwargs
        if not self.text_field_kwargs:
            self.text_field_kwargs = DEFAULT_TEXT_FIELD_KWARGS
        self.label_field_kwargs = label_field_kwargs
        if not self.label_field_kwargs:
            self.label_field_kwargs = DEFAULT_LABEL_FIELD_KWARGS

        self.validate()
        self.set_embedding_vectors_cfg()

    def set_embedding_vectors_cfg(self):
        if self.embedding_dim == 50:
            self.embedding_vectors_cfg = 'glove.6B.50d'
        elif self.embedding_dim == 100:
            self.embedding_vectors_cfg = 'glove.6B.100d'
        elif self.embedding_dim == 200:
            self.embedding_vectors_cfg = 'glove.6B.200d'
        elif self.embedding_dim == 300:
            if self.num_tokens_embedding_train == '6B':
                self.embedding_vectors_cfg = 'glove.6B.300d'
            elif self.num_tokens_embedding_train == '42B':
                self.embedding_vectors_cfg = 'glove.42B.300d'
            elif self.num_tokens_embedding_train == '840B':
                self.embedding_vectors_cfg = 'glove.840B.300d'

    def validate(self):
        if not isinstance(self.max_vocab_size, int):
            msg = "max_vocab_size must be an integer"
            logger.error(msg)
            raise ValueError(msg)
        if self.max_vocab_size < 1:
            msg = "max_vocab_size must be atleast 1!"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.embedding_dim, int):
            msg = "embedding_dim must be an integer"
            logger.error(msg)
            raise ValueError(msg)
        if self.embedding_dim < 1:
            msg = "embedding dimension must be atleast 1"
            logger.error(msg)
            raise ValueError(msg)
        if self.embedding_type not in ['glove']:
            msg = "embedding type must be one of: ['glove']"
            logger.error(msg)
            raise ValueError(msg)
        if self.embedding_dim not in [50, 100, 200, 300]:
            msg = "embedding dimension must be one of [50, 100, 200, 300]"
            logger.error(msg)
            raise ValueError(msg)
        if self.num_tokens_embedding_train not in ['6B', '42B', '840B']:
            msg = "number of tokens to train the embedding must be one of: ['6B', '42B', '840B']"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.text_field_kwargs, dict):
            msg = "text_field_kwargs must be a dictionary used to configure the torchtext.data.Field object!"
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.label_field_kwargs, dict):
            msg = "text_field_kwargs must be a dictionary used to configure the torchtext.data.LabelField object!"
            logger.error(msg)
            raise ValueError(msg)


class ImageDataConfiguration(DataConfiguration):
    pass
