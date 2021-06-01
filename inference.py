from models.transformers import TransformerEncoder, TransformerDecoder, Seq2Seq
from models.gru_encoder_decoder_attention import Encoder as GRU_Encoder, Decoder as GRU_Decoder, Seq2Seq as GRU_Seq2Seq

import torch
from models.utils.beam import Beam, GNMTGlobalScorer, _from_beam
from models.utils.beam_attention import BeamAttention, _from_beam_attention
from models.utils.beam_n_gram import beam_search_n_gram
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
from models.utils.text_processing import clean_text

PAD_IDX, BOS_IDX, EOS_IDX = 1, 2, 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def transformers_evaluate(net, sentence, no_tone_vocab, tone_vocab, max_length, device, beam_size):
  net.eval()

  src_tokens = [no_tone_vocab.stoi[word] for word in sentence.lower().split(" ")] + [EOS_IDX]

  # print(src_tokens)
  # unsqueeze add more axis (add batch axis)
  enc_X = torch.unsqueeze(
      torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
  )

  # create mask for encoder
  # mask: (batch_size*num_heads x 1 x num_steps)
  mask = (enc_X == PAD_IDX).unsqueeze(1).type(torch.bool).repeat_interleave(net.encoder.num_heads, dim=0).to(device)
  with torch.no_grad():
    enc_outputs = net.encoder(enc_X, mask)
    # enc_outputs: (batch_size x num_steps x num_hiddens)
  
  
  dec_state = net.decoder.init_state(enc_outputs, mask)#.repeat((1, beam_size, 1))
  # dec_state: [enc_outputs, mask, [None]*self.num_layers]
  # [(batch_size x num_steps x num_hiddens), (batch_size*num_heads x num_steps), [None]*self.num_layers]

  dec_state[0] = dec_state[0].repeat((beam_size, 1, 1))
  dec_state[1] = dec_state[1].repeat((beam_size, 1, 1))

  beam = [Beam(beam_size, 1, 2, 3, GNMTGlobalScorer(), 5) for _ in range(1)]

  for i in range(max_length):
    if all((b.done() for b in beam)):
      break
    with torch.no_grad():
      # beam search automatically return <bos> in the first time
      dec_X = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1).t().to(device)
      Y, dec_state = net.decoder(dec_X, dec_state)

      select_indices_array = []
      for j, b in enumerate(beam):
        b.advance(Y[:, j])

  ret = _from_beam(beam)
  return ret

def gru_evaluate(net, sentence, no_tone_vocab, tone_vocab, max_length, device, beam_size):
  net.eval()

  src_tokens = [no_tone_vocab.stoi[word] for word in sentence.lower().split(" ")] + [no_tone_vocab.stoi['<eos>']]

  # unsqueeze add batch dim
  enc_X = torch.unsqueeze(
      torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
  )
  mask = (enc_X == no_tone_vocab.stoi['<pad>']).unsqueeze(1)

  #TODO: add attention to beam
  net.decoder._attention_weights = []

  with torch.no_grad():
    enc_outputs = net.encoder(enc_X)
    # enc_outputs
    # output: (num_steps, batch_size, num_hiddens)
    # state : (num_layers, batch_size, num_hiddens)
  
  outputs, dec_state = net.decoder.init_state(enc_outputs)
  # outputs:   (batch_size, num_steps, num_hiddens)
  # dec_state: (num_layers, batch_size, num_hiddens)
  
  outputs = outputs.repeat((beam_size, 1, 1))
  dec_state = dec_state.repeat((1, beam_size, 1))


  beam = [BeamAttention(beam_size, 1, 2, 3, GNMTGlobalScorer(), 5) for _ in range(1)]
  

  for i in range(max_length):
    if all((b.done() for b in beam)):
      break
    with torch.no_grad():
      dec_X = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1).t().to(device)
      Y, (outputs, dec_state) = net.decoder(dec_X, (outputs, dec_state), mask)
      # Y:        (batch_size, num_steps, vocab_size)
      # outputs:  (batch_size, num_steps, num_hiddens)
      # dec_state:(num_layers, batch_size, num_hiddens)

      select_indices_array = []
      for j, b in enumerate(beam):
        attn = torch.cat([steps for steps in net.decoder._attention_weights], dim=1)
        b.advance(Y[:, j], attn)
        select_indices_array.append(b.get_current_origin() * 1 + j)
      
      select_indices = torch.cat(select_indices_array) \
                      .view(1, beam_size) \
                      .transpose(0, 1) \
                      .contiguous() \
                      .view(-1).type(torch.int32).to(device)
      
      dec_state = dec_state.index_select(1, select_indices)
  ret = _from_beam_attention(beam)
  return ret

def n_gram_evaluate(ngram_model, sentence):
  return beam_search_n_gram(sentence.lower().split(), ngram_model)

tone_vocab = torch.load('./data/vocab/tone_vocab.vocab')
no_tone_vocab = torch.load('./data/vocab/no_tone_vocab.vocab')

# TODO: ...
tone_vocab_size = len(tone_vocab)
no_tone_vocab_size = len(no_tone_vocab)

# transformer model
num_hiddens, num_layers, dropout, batch_size = 512, 6, 0.0, 512

ffn_num_input, ffn_num_hiddens, num_heads = 512, 1024, 8
key_size, query_size, value_size = 512, 512, 512
norm_shape = [512]

enc = TransformerEncoder(no_tone_vocab_size, key_size, query_size, value_size,
                          num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

dec = TransformerDecoder(tone_vocab_size, key_size, query_size, value_size,
                          num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

transformer_net = Seq2Seq(enc, dec, PAD_IDX, device).to(device)

transformer_net.load_state_dict(torch.load("./models/weights/transformers_weight.pt", map_location=torch.device('cpu')))
transformer_net.eval()
print('Transformers is loaded')

# gru encoder decoder
embed_size = 128
num_hiddens = 128
num_layers = 1
dropout = 0.0

lr = 0.008
num_epochs = 20

gru_enc = GRU_Encoder(no_tone_vocab_size, embed_size, num_hiddens, num_layers, dropout)

gru_dec = GRU_Decoder(tone_vocab_size, embed_size, num_hiddens, num_layers, dropout)

gru_net = GRU_Seq2Seq(gru_enc, gru_dec, tone_vocab.stoi['<pad>'], device).to(device)

gru_net.load_state_dict(torch.load("./models/weights/gru_encoder_decoder_attention_weights.pth", map_location=torch.device('cpu')))
gru_net.eval()
print('GRU Encoder Decoder is loaded')

# n-gram model

ngram_model_dir = "models/weights"
with open(os.path.join(ngram_model_dir, 'n-gram.pkl'), 'rb') as fin:
  ngram_model = pickle.load(fin)

detokenize = TreebankWordDetokenizer().detokenize
print('N-gram model is loaded')

# print(len(model_loaded.vocab))



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/transformers_predict',methods=['POST'])
@cross_origin()
def transformer_predict():

    data = request.get_json(force=True)
    sentence = clean_text(data['sentence'])
    
    beam = transformers_evaluate(transformer_net, sentence, no_tone_vocab, tone_vocab, 20, device, 4)
    
    text_sentence = []

    attention = [[head_attention.tolist() for head_attention in layer_attention] for layer_attention in transformer_net.encoder.attention_weights]

    for num_predict in range(len(beam['predictions'])):
        for prediction in beam['predictions'][num_predict]:
          text_sentence.append(' '.join(tone_vocab.itos[idx] for idx in prediction[:-1]))
    
    return jsonify({"text_sentence": text_sentence, 'attention': attention})


@app.route('/gru_predict',methods=['POST'])
@cross_origin()
def gru_predict():

    data = request.get_json(force=True)
    sentence = clean_text(data['sentence'])
    
    beam = gru_evaluate(gru_net, sentence, no_tone_vocab, tone_vocab, 20, device, 4)
    
    text_sentence = []

    attention = []
    for beam_text in beam['attention'][0]:
      steps_attention = []
      for steps in beam_text[::-1]:
        steps_attention.append(steps[-1].tolist())
      attention.append(steps_attention)
    # print(attention[0].shape)
    # print(len(attention))

    for num_predict in range(len(beam['predictions'])):
        for prediction in beam['predictions'][num_predict]:
          text_sentence.append(' '.join(tone_vocab.itos[idx] for idx in prediction[:-1]))
    
    return jsonify({"text_sentence": text_sentence, 'attention': attention})

@app.route('/n_gram_predict',methods=['POST'])
@cross_origin()
def n_gram_predict():

    data = request.get_json(force=True)
    sentence = clean_text(data['sentence'])
    
    result = n_gram_evaluate(ngram_model, sentence)
    text_sentence = [' '.join(w for w in r[0]) for r in result]
    score = [r[1] for r in result]
    
    return jsonify({"text_sentence": text_sentence, 'score': score})