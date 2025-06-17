import torch
from torch import nn

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        outputs, h_n = self.gru(embedded)     # h_n: (num_layers * num_directions, batch, hidden_size)
        return outputs, h_n[-1]               # use last layer's hidden state (batch, hidden_size)
