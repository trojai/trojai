import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from transformers import BertTokenizer, BertModel


class EmbeddingLSTM(nn.Module):
    """
    Defines an LSTM model that can be used for text classification
    from: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment
            %20Analysis.ipynb
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        logger.info("Creating EmbeddingLSTM model with Embedding [" + str(vocab_size) +
                    " x " + str(embedding_dim) + "]. pad_idx=" + str(pad_idx))
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.packed_padded_sequences = True

    def forward(self, text, text_lengths):
        # text.shape = [sentence len, batch size]
        # embedded.shape = [sentence len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # hidden.shape = [num layers * num directions, batch size, hid dim]
        # cell.shape = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden.shape  = [batch size, hid dim * num directions]

        return self.fc(hidden)


class EmbeddingGRU(nn.Module):
    """
    Defines an GRU model that can be used for text classification
    from: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment
            %20Analysis.ipynb
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        logger.info("Creating EmbeddingGRU model with Embedding [" + str(vocab_size) +
                    " x " + str(embedding_dim) + "]. pad_idx=" + str(pad_idx))
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.packed_padded_sequences = True

    def forward(self, text, text_lengths):
        # text.shape = [sentence len, batch size]
        # embedded.shape = [sentence len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # hidden.shape = [num layers * num directions, batch size, hid dim]
        # cell.shape = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden.shape  = [batch size, hid dim * num directions]

        return self.fc(hidden)


class XFormerGRU(nn.Module):
    """
    From: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb
    """
    def __init__(self, xformer, hidden_dim, output_dim, n_layers, bidirectional, dropout, freeze_bert=True):
        super().__init__()
        self.xformer = xformer
        embedding_dim = xformer.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.packed_padded_sequences = False

        if freeze_bert:
            for name, param in self.named_parameters():
                if name.startswith('xformer'):
                    param.requires_grad = False

    def forward(self, text):

        # text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.xformer(text)[0]

        # embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.out(hidden)
        # output = [batch size, out dim]

        return output


def pretrained_bert_gru_sentiment_model(hidden_dim, output_dim, n_layers, bidirectional, dropout, freeze_bert=True):
    """

    :return:
    """
    bert = BertModel.from_pretrained('bert-base-uncased')

    # build the model
    model = XFormerGRU(bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, freeze_bert=freeze_bert)

    return model


def bert_gru_sentiment_model(hidden_dim, output_dim, n_layers, bidirectional, dropout, **kwargs):
    pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
    bert_config = copy.copy(pretrained_bert.config)
    # NOTE: I don't think a deep copy is needed here because the BertConfig object doesn't
    #  have any compound objects.
    #  See here: https://huggingface.co/transformers/_modules/transformers/configuration_bert.html#BertConfig
    pretrained_bert = None  # try to get the garbage collector to clean this up from memory

    # TODO: update this to be more efficient (i.e. get the config directly from the pretrained function rather than
    #  instantiating an object and then pulling hte ocnfig from that) - for now, we are just testing things out
    #  I think this is the config: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    raw_bert = BertModel(bert_config)
    # build the model
    model = XFormerGRU(raw_bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, freeze_bert=False)

    return model


def bert_tokenizer():
    # The vocabulary for this tokenizer is here:
    #  https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return bert_tokenizer