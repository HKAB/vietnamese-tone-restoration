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
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotProductAttention(nn.Module):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

  # queries:  (batch_size*num_head x num_steps x num_hiddens/num_head) 
  # keys:     (batch_size*num_head x num_steps x num_hiddens/num_head) 
  # values:   (batch_size*num_head x num_steps x num_hiddens/num_head) 
  # mask:     (batch_size*num_heads x 1 x num_steps)
  def forward(self, queries, keys, values, mask):
    d = queries.shape[-1]
    scores = torch.bmm(queries, keys.transpose(1, 2))/math.sqrt(d)

    # scores: (batch_size*num_head x num_steps x num_steps)
    # mask:   (batch_size*num_heads x 1 x num_steps)
    # self.attention_weights: (batch_size*num_head x num_steps x num_steps)
    self.attention_weights = F.softmax(scores.masked_fill(mask=mask, value=-np.inf), dim=-1)

    # output: (batch_size*num_head x num_steps x num_hiddens/num_head)
    return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
  # X: batch_size x key-value pairs x num_hiddens

  # X shape -> batch_size x key-value pairs x num_head x num_hiddens/num_head
  X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
  X = X.permute(0, 2, 1, 3)

  return X.reshape(-1, X.shape[2], X.shape[3])
def transpose_output(X, num_heads):
  X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

  X = X.permute(0, 2, 1, 3)

  return X.reshape(X.shape[0], X.shape[1], -1)

class MultiheadAttention(nn.Module):
  def __init__(self, 
               key_size, query_size, value_size, 
               num_hiddens, num_heads, dropout, 
               bias=False):
    super().__init__()
    self.num_heads = num_heads
    # self.attention = AdditiveAttention(int(key_size/num_heads), int(query_size/num_heads), int(num_hiddens/num_heads), dropout)
    self.attention = DotProductAttention(dropout)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
    self.W_v = nn.Linear(value_size, num_hiddens, bias=False)
    self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)
  
  # queries: (batch_size x num_steps x num_hiddens) 
  # keys: (batch_size x num_steps x num_hiddens) 
  # values: (batch_size x num_steps x num_hiddens) 
  # mask: (batch_size*num_heads x num_steps)
  def forward(self, queries, keys, values, mask):
    queries = transpose_qkv(self.W_q(queries), self.num_heads)
    keys = transpose_qkv(self.W_k(keys), self.num_heads)
    values = transpose_qkv(self.W_v(values), self.num_heads)
    
    # after transpose_qkv
    # queries: (batch_size*num_head x num_steps x num_hiddens/num_head) 
    # keys: (batch_size*num_head x num_steps x num_hiddens/num_head) 
    # values: (batch_size*num_head x num_steps x num_hiddens/num_head) 

    output = self.attention(queries, keys, values, mask)
    # output: (batch_size*num_head x num_steps x num_hiddens/num_head)

    output_concat = transpose_output(output, self.num_heads)
    # output_concat: (batch_size x num_steps x num_hiddens)
    # print(output_concat.shape, values.shape)
    # return shape: (batch_size x num_steps x num_hiddens)
    return self.W_o(output_concat)

class PositionalEncoding(nn.Module):
  def __init__(self, num_hiddens, dropout, max_len=1000):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

    self.P = torch.zeros((1, max_len, num_hiddens))

    X = torch.arange(max_len, dtype=torch.float32).reshape(
        -1, 1)/torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32)/num_hiddens)
    self.P[:, :, 0::2] = torch.sin(X)
    self.P[:, :, 1::2] = torch.cos(X)
  
  # X: (batch_size x num_steps x num_hiddens)
  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :].to(X.device)
    # return shape: (batch_size x num_steps x num_hiddens)
    return self.dropout(X)

class PositionWiseFFN(nn.Module):
  def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs):
    super().__init__()
    self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
    self.relu = nn.ReLU()
    self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
  # X: (batch_size x num_steps x num_hiddens)
  def forward(self, X):
    # FFN is the same for every num_steps
    # num_hiddens should be equal to ffn_num_inputs, ffn_num_outputs
    # self.dense1(X): (batch_size x num_steps x ffn_num_hiddens)
    # return shape:   (batch_size x num_steps x ffn_num_outputs)
    return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
  def __init__(self, normalized_shape, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(normalized_shape)
  # X: (batch_size x num_steps x num_hiddens)
  # Y: (batch_size x num_steps x num_hiddens)
  def forward(self, X, Y):
    # return shape: (batch_size x num_steps x num_hiddens)
    return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False):
    super().__init__()
    self.attention = MultiheadAttention(key_size, query_size, value_size, 
                                        num_hiddens, num_heads, dropout)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm2 = AddNorm(norm_shape, dropout)
  # X: (batch_size x num_steps x num_hiddens) 
  # mask: (batch_size*num_heads x num_steps)
  def forward(self, X, mask):
    # self.attention(X, X, X, mask): (batch_size x num_steps x num_hiddens)
    Y = self.addnorm1(X, self.attention(X, X, X, mask))
    # Y: (batch_size x num_steps x num_hiddens)

    # self.ffn(Y):  (batch_size x num_steps x num_hiddens)
    # return shape: (batch_size x num_steps x num_hiddens)
    return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
  def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False):
    super().__init__()
    self.num_hiddens = num_hiddens
    self.num_heads = num_heads
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()

    for i in range(num_layers):
      self.blks.add_module(
          "block " + str(i),
          EncoderBlock(key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias))
  # X: (batch_size x num_steps)
  # mask: (batch_size*num_heads x 1 x  num_steps)
  def forward(self, X, mask):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    # X: (batch_size x num_steps x num_hiddens)

    self.attention_weights = [None]*len(self.blks)

    for i, blk in enumerate(self.blks):
      # X: (batch_size x num_steps x num_hiddens)
      X = blk(X, mask)
      self.attention_weights[i] = blk.attention.attention.attention_weights

    # return shape: (batch_size x num_steps x num_hiddens)
    return X

class DecoderBlock(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i):
    super().__init__()
    self.i = i
    self.num_heads = num_heads
    self.attention1 = MultiheadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    # self.attention1 = DotProductAttention(dropout)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.attention2 = MultiheadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm2 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm3 = AddNorm(norm_shape, dropout)


  # X:                 (batch_size x num_steps, num_hiddens)
  # state: [(batch_size x num_steps x num_hiddens), (batch_size*num_heads x num_steps), [None]*self.num_layers]
  def forward(self, X, state):
    enc_outputs, enc_mask = state[0], state[1]
    # enc_outputs: (batch_size x num_steps x num_hiddens)
    # enc_mask:    (batch_size*num_heads x num_steps)

    if state[2][self.i] is None:
      key_values = X
      # key_values: (batch_size x num_steps x num_hiddens)
    else:
      key_values = torch.cat([state[2][self.i], X], axis = 1)
      # never run in here?
      # print("layer", self.i, "key_values.shape", key_values.shape)
    state[2][self.i] = key_values

    batch_size, num_steps, _ = X.shape
    if self.training:
      # spectial mask for each time step
      dec_mask = torch.ones(batch_size, num_steps, num_steps).triu(diagonal=1).type(torch.bool).repeat(self.num_heads, 1, 1).to(device)
      # dec_mask = torch.zeros(batch_size, num_steps).type(torch.bool).unsqueeze(1)#.repeat(self.num_heads, 1, 1).to(device)
      # dec_mask: (batch_size*num_heads, 1, num_steps)
    else:
      dec_mask = torch.zeros(batch_size, num_steps, num_steps).type(torch.bool).repeat(self.num_heads, 1, 1).to(device)
      # dec_mask: (batch_size*num_heads, 1, num_steps)
    
    # self attention?
    X2 = self.attention1(X, key_values, key_values, dec_mask)
    # X2: (batch_size x num_steps x num_hiddens)
    
    Y = self.addnorm1(X, X2)
    # Y: (batch_size x num_steps x num_hiddens)

    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_mask)
    # Y2: (batch_size x num_steps x num_hiddens)
    Z = self.addnorm2(Y, Y2)
    # Z: (batch_size x num_steps x num_hiddens)

    output = self.addnorm3(Z, self.ffn(Z))
    # output: (batch_size x num_steps x num_hiddens)
    # state[2][self.i] is updated
    return output, state

class TransformerDecoder(nn.Module):
  def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout):
    super().__init__()
    self.num_hiddens = num_hiddens
    self.num_layers = num_layers

    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module(
          'block' + str(i),
          DecoderBlock(key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i)
      )
      self.dense = nn.Linear(num_hiddens, vocab_size)
  def init_state(self, enc_outputs, mask):
    return [enc_outputs, mask, [None]*self.num_layers]

  # X    : (batch_size x num_steps)
  # state: [(batch_size x num_steps x num_hiddens), (batch_size*num_heads x num_steps), [None]*self.num_layers]
  def forward(self, X, state):
    
    X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
    # self.embedding(X): (batch_size x num_steps, num_hiddens)
    # X:                 (batch_size x num_steps, num_hiddens)
    
    self._attention_weights = [[None] * len(self.blks) for _ in range(2)]

    for i, blk in enumerate(self.blks):
      X, state = blk(X, state)
      # X: (batch_size x num_steps x num_hiddens)
      # [(batch_size x num_steps x num_hiddens), (batch_size*num_heads x num_steps), [(state)]]
      self._attention_weights[0][i] = blk.attention1.attention.attention_weights
      self._attention_weights[1][i] = blk.attention2.attention.attention_weights
    
    # return shape: (batch_size x num_steps x vocab_size), [(state)]
    return self.dense(X), state
  def attention_weights(self):
    return self._attention_weights

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
    def create_mask(self, src, num_heads):
      return (src == self.pad_idx).unsqueeze(1).type(torch.bool).repeat_interleave(num_heads, dim=0).to(device)
    # enc_X: (batch_size x num_steps)
    # dec_X: (batch_size x num_steps)
    def forward(self,
                enc_X: Tensor,
                dec_X: Tensor) -> Tensor:
        mask = self.create_mask(enc_X, self.encoder.num_heads).to(self.device)
        # mask: (batch_size*num_heads x 1 x num_steps)

        enc_outputs = self.encoder(enc_X, mask)
        # enc_outputs: (batch_size x num_steps x num_hiddens)

        dec_state = self.decoder.init_state(enc_outputs, mask)
        # dec_state: [enc_outputs, mask, [None]*self.num_layers]
        # [(batch_size x num_steps x num_hiddens), (batch_size*num_heads x num_steps), [None]*self.num_layers]

        # return shape: (batch_size x num_steps x vocab_size)
        return self.decoder(dec_X, dec_state)