import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
