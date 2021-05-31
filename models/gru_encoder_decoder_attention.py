import torch
import torchtext
from torchtext.data import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
        
        self.dropout = nn.Dropout(dropout)

    # X: (batch_size x num_steps)

    def forward(self,
                X: Tensor) -> Tuple[Tensor]:
        X = self.dropout(self.embedding(X))
        # X: (batch_size, num_steps, embed_size)

        X = X.permute(1, 0, 2)
        # X: (num_steps, batch_size, embed_size)
        
        output, state = self.rnn(X)
        # output: (num_steps, batch_size, num_hiddens)
        # state : (num_layers, batch_size, num_hiddens)
        
        return output, state
class AdditiveAttention(nn.Module):
  def __init__(self, key_size, query_size, num_hiddens, dropout):
    super().__init__()
    self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
    self.W_v = nn.Linear(num_hiddens, 1, bias=False)
    self.dropout = nn.Dropout(dropout)

  # query:  (batch_size x 1 x num_hiddens)
  # keys:   (batch_size, num_steps, num_hiddens)
  # values: (batch_size, num_steps, num_hiddens)
  # mask:   (batch_size x 1 x num_steps)
  def forward(self, queries, keys, values, mask):
    queries, keys = self.W_q(queries), self.W_k(keys)
    # query:  (batch_size x 1 x num_hiddens)
    # keys:   (batch_size, num_steps, num_hiddens)
    # i choosed key_size = num_hiddens

    # queries: (batch_size x 1 x 1 x num_hiddens) `1 query`
    # keys:    (batch_size x 1 x num_steps x num_hiddens)
    # broadcast sum
    features = queries.unsqueeze(2) + keys.unsqueeze(1)
    # features: (batch_size x 1 x num_steps x num_hiddens)

    features = torch.tanh(features)

    scores = self.W_v(features).squeeze(-1)
    # scores: (batch_size x 1 x num_steps x 1).squeeze(-1): (batch_size x 1 x num_steps)

    self._attention_weights = F.softmax(scores.masked_fill(mask=mask, value=-np.inf), dim=-1)
    
    # return shape: (batch_size, 1, num_hiddens) `1 query`
    return torch.bmm(self._attention_weights, values)


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout)

        self._attention_weights = []
    
    def init_state(self, enc_outputs):
      # outputs (num_steps x batch_size x num_hidden)
      outputs, hidden_states = enc_outputs
      return (outputs.permute(1, 0, 2), hidden_states)
    
    # X:     (batch_size x num_steps)
    # state: ((batch_size, num_steps, num_hiddens), (num_layers, batch_size, num_hiddens))
    # mask:  (batch_size x 1 x num_steps)
    def forward(self,
                X: Tensor,
                state: Tensor,
                mask: Tensor) -> Tuple[Tensor]:
        X = self.dropout(self.embedding(X))
        # X: (batch_size x num_steps x embed_size)

        X = X.permute(1, 0, 2)
        # X: (num_steps x batch_size x embed_size)

        enc_outputs, hidden_state = state

        outputs = []

        # for every steps
        for x in X:
          query = torch.unsqueeze(hidden_state[-1], dim=1)
          # query: (batch_size x 1 x num_hiddens)
          context = self.attention(query, enc_outputs, enc_outputs, mask)
          # context: (batch_size, 1, num_hiddens)
          x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
          # x = (batch_size, 1, num_hiddens) cat with (batch_size x 1 x embed_size)
          # x: (batch_size, 1, num_hiddens + embed_size)
          output, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
          # output      : (1, batch_size, num_hiddens + embed_size)
          # hidden_state: (num_layers, batch_size, num_hiddens)

          hidden_state.detach()
          
          outputs.append(output)
          self._attention_weights.append(self.attention._attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        outputs = self.log_softmax(outputs)

        # return shape: (batch_size, num_steps, vocab_size), [(batch_size, num_steps, num_hiddens), (num_layers, batch_size, num_hiddens)]
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state]


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 pad_idx: int,
                 device: torch.device,):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
    def create_mask(self, src):
      return (src == self.pad_idx).unsqueeze(1)
    
    # enc_x: (batch_size x num_steps)
    # dec_X: (batch_size x num_steps)
    def forward(self,
                enc_X: Tensor,
                dec_X: Tensor) -> Tensor:
        mask = self.create_mask(enc_X)
        # mask: (batch_size x 1 x num_steps)

        enc_outputs = self.encoder(enc_X)
        # enc_outputs
        # output: (num_steps, batch_size, num_hiddens)
        # state : (num_layers, batch_size, num_hiddens)

        dec_state = self.decoder.init_state(enc_outputs)
        # dec_state: ((batch_size, num_steps, num_hiddens), (num_layers, batch_size, num_hiddens))

        return self.decoder(dec_X, dec_state, mask)

