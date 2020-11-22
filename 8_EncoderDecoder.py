# coding:utf-8

import re
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import random
import codecs
import pathlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'
PUNCTUATIONS = r"[!\"#$%&'()\*\+,-./:;<=>\?@\[\]^_`{\|}~“”？，！【】（）、。：；’‘……￥·]"

### Preprocess the data ###
class DataProcess(object):
  def __init__(self, data_path: pathlib.PosixPath) -> None:
    # read the data
    with codecs.open(data_path, 'r', 'utf-8') as file:
      self.datas = file.read().split('\n')[:-1]
    self.data_length = len(self.datas)
    print('The data contains {} lines.'.format(self.data_length))

    # preprocess the data
    self.eng_datas = []
    self.chs_datas = []
    for line in self.datas:
      line = line.split('\t')
      self.eng_datas.append(self._cleanLine(line[0]))
      self.chs_datas.append(self._cleanLine(line[1]))
    
    # build vocab dict
    self.eng_vocab_idx, self.eng_idx_vocab = self._buildDict(self.eng_datas, ' ')
    self.chs_vocab_idx, self.chs_idx_vocab = self._buildDict(self.chs_datas, '')
    print('English Words Number: {}\t Chinese Words Number: {}'\
              .format(len(self.eng_vocab_idx), len(self.chs_vocab_idx)))

  def _cleanLine(self, line: str) -> str:
    """Clean the line by upperdown the letters."""
    line = re.sub(PUNCTUATIONS , '', line.lower().strip())
    return line

  def _buildDict(self, datas: list, split_tag: str) -> tuple:
    """Build vocab for datas."""
    vocab_idx = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
    idx_vocab = {0: SOS, 1: EOS, 2: PAD, 3: UNK}
    
    for line in datas:
      for word in line if split_tag == '' else line.split(split_tag):
        if word not in vocab_idx:
          vocab_idx[word] = len(vocab_idx)
          idx_vocab[len(idx_vocab)] = word
    print('There are {} words.'.format(len(vocab_idx)))

    return vocab_idx, idx_vocab
  
  def _createBatchIndices(self, data_length, batch_size):
    """Create batch indices according to the whole data."""
    batch_number = data_length // batch_size
    reminder = data_length % batch_size
    batch_number = batch_number if reminder == 0 else batch_number + 1

    for idx in range(batch_number):
      yield (idx * batch_size, idx * batch_size + batch_size)

  def _padding(self, data, pad_idx):
    seq_length = [len(line) for line in data]
    max_length = max(seq_length)

    padding_func = lambda line : line + [pad_idx for _ in range(max_length - len(line))]
    data_padded = list(map(padding_func, data))

    return data_padded, seq_length
  
  def _convertToIdx(self, line, vocab_idx, split_tag):
    line = line if split_tag == '' else line.split(split_tag)
    return [vocab_idx[word] if word in vocab_idx else vocab_idx[UNK] for word in line]
  
  def convertToStr(self, line, idx_vocab):
    if type(line) == torch.Tensor:
      return [idx_vocab[idx.item()] if idx.item() in idx_vocab else idx_vocab[3] for idx in line]
    else:
      return [idx_vocab[idx] if idx in idx_vocab else idx_vocab[3] for idx in line]

  def _add_tag(self, line, tag, is_front):
    return [tag] + line if is_front else line + [tag]

  def dataBatch(self, batch_size: int, is_shuffle: bool=True) -> dict:
    """Provide the batch data."""
    data_pairs = list(zip(self.eng_datas, self.chs_datas))
    if is_shuffle:
      random.shuffle(data_pairs)
    eng_datas, chs_datas = zip(*data_pairs)

    for start, end in self._createBatchIndices(self.data_length, batch_size):
      # 'temp' indicates that '<sos>', '<eos>' and '<pad>' will be added.
      scr_temp_data = eng_datas[start : end]
      tgt_temp_data = chs_datas[start : end]

      # convert str to idx
      convert_src_func = partial(self._convertToIdx, vocab_idx=self.eng_vocab_idx, split_tag=' ')
      scr_temp_data = list(map(convert_src_func, scr_temp_data))
      convert_tgt_func = partial(self._convertToIdx, vocab_idx=self.chs_vocab_idx, split_tag='')
      tgt_temp_data = list(map(convert_tgt_func, tgt_temp_data))

      # add <sos>, <eos> to target data
      add_sos_func = partial(self._add_tag, tag=self.chs_vocab_idx[SOS], is_front=True)
      tgt_temp_input_data = list(map(add_sos_func, tgt_temp_data))
      add_eos_func = partial(self._add_tag, tag=self.chs_vocab_idx[EOS], is_front=False)
      tgt_temp_output_data = list(map(add_eos_func, tgt_temp_data))

      # padding the idx
      scr_input_data, src_seq_length = self._padding(scr_temp_data, self.eng_vocab_idx[PAD])
      tgt_input_data, _ = self._padding(tgt_temp_input_data, self.chs_vocab_idx[PAD])
      tgt_output_data, tgt_seq_length = self._padding(tgt_temp_output_data, self.chs_vocab_idx[PAD])

      data_batch = {'src_input_data': torch.tensor(scr_input_data),
                    'src_seq_length': torch.tensor(src_seq_length),
                    'tgt_input_data': torch.tensor(tgt_input_data),
                    'tgt_output_data': torch.tensor(tgt_output_data),
                    'tgt_seq_length': torch.tensor(tgt_seq_length),
                    'actual_bs': len(scr_input_data)}
      yield data_batch

### Model ###
class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super(Encoder, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(input_size=hidden_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=True,
                      batch_first=True)
    self.out = nn.Linear(hidden_size * 2, hidden_size)
  
  def forward(self, src_input_data, src_seq_length, hidden):
    """Forward the Bi-directional GRU.
    Args:
      src_input: [batch_size, seq_length].
      hidden: [num_layers*2, batch_size, hidden]
    Returns:
      output: [batch_size, src_seq_length, hidden]
    """
    # Embedding: (b, s) -> (b, s, h)
    embedded = self.embedding(src_input_data)

    # GRU
    padded_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, 
                                                              src_seq_length,
                                                              batch_first=True,
                                                              enforce_sorted=False)
    # output: (b, s, 2h), hidden: (2n, b, h)
    output, hidden = self.gru(padded_embedded, hidden)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

    # (b, s, 2h) -> (b, s, h)
    output = self.out(output)

    return output
  
  def initHidden(self, batch_size):
    return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)

class Decoder(nn.Module):
  def __init__(self, hidden_size, output_size, num_layers=2, dropout=0.1):
    super(Decoder, self).__init__()

    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.dropout = dropout

    self.embedding = nn.Embedding(output_size, hidden_size)
    self.query_matrix = nn.Linear(hidden_size, hidden_size)
    self.key_matrix = nn.Linear(hidden_size, hidden_size)
    self.value_matrix = nn.Linear(hidden_size, hidden_size)

    self.dropout = nn.Dropout(dropout)

    self.gru = nn.GRU(input_size=hidden_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=True,
                      batch_first=True)
    
    self.out = nn.Linear(hidden_size*2, output_size)
  
  def forward(self, input_word, hidden, encoder_outputs, attention_mask):
    """
      Args:
        input_word: [batch_size, 1].
        hidden: [num_layers*2, batch_size, hidden_size].
        encoder_outputs: [batch_size, src_seq_length, hidden_size].
        scr_seq_length: [batch_size]
      Returns:
        output: [batch_size, output_size]
    """
    # embedding: (b, 1) -> (b, 1, h)
    embedded = self.embedding(input_word)
    embedded = self.dropout(embedded)

    # query: (b, 1, h)
    query = self.query_matrix(embedded)
    # key: (b, src_seq_length, hidden_size) -> (b, hidden_size, src_seq_length)
    key = self.key_matrix(encoder_outputs)
    key = key.permute(0, 2, 1)
    # value: (b, src_seq_length, hidden_size)
    value = self.value_matrix(encoder_outputs)
    # attention_scores: (b, 1, src_seq_length)
    attention_scores = torch.matmul(query, key)
    attention_scores += attention_mask
    attention_probs = F.softmax(attention_scores, dim=-1)

    # context_layer (b, 1, h), no need to squeeze the second dimension,
    # as the second dimension is time step.
    context_layer = F.relu(torch.matmul(attention_probs, value))
    output, hidden = self.gru(context_layer, hidden)
    # output: (b, o)
    output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
    return output, hidden
  
  def initHidden(self, batch_size):
    return torch.zeros(self.num_layers*2, batch_size, self.hidden_size)

def makeAttentionMask(src_seq_length):
  max_length = torch.max(src_seq_length)
  attention_mask = torch.arange(max_length)[None, :] < src_seq_length[:, None]
  attention_mask = attention_mask.float()[:, None, :]
  attention_mask = (attention_mask - 1) * 10000
  return attention_mask

def train(datas, optimizers, is_predict=False, predict_path='prediciton.txt', sample_k=3):
  src_input_data = datas['src_input_data']
  src_seq_length = datas['src_seq_length']
  tgt_input_data = datas['tgt_input_data']
  tgt_output_data = datas['tgt_output_data']
  tgt_seq_length = datas['tgt_seq_length']
  batch_size = datas['actual_bs']
  
  src_sentences = [data_process.convertToStr(line, data_process.eng_idx_vocab) for line in src_input_data]
  tgt_sentences = [data_process.convertToStr(line, data_process.chs_idx_vocab) for line in tgt_output_data]
  
  encoder_optimizer = optimizers['encoder']
  decoder_optimizer = optimizers['decoder']
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Encoder
  encoder_hidden = encoder.initHidden(batch_size)
  encoder_outputs = encoder(src_input_data, src_seq_length, encoder_hidden)

  # Decoder
  attention_mask = makeAttentionMask(src_seq_length)
  decoder_hidden = decoder.initHidden(batch_size)
  seq_length = tgt_input_data.size()[1]
  tgt_loss_mask = torch.arange(torch.max(tgt_seq_length))[None, :] < tgt_seq_length[:, None]
  tgt_loss_mask = tgt_loss_mask.float()

  loss = 0
  predictions = []
  for t in range(seq_length):
    # (b, 1)
    input_t = tgt_input_data[:, t][:, None]
    golden_t = tgt_output_data[:, t]
    mask_t = tgt_loss_mask[:, t]

    output, decoder_hidden = decoder(input_t, decoder_hidden, encoder_outputs, attention_mask)
    topv, topi = output.topk(1)
    predictions.append(topi)
    loss += torch.sum(criterion(output, golden_t) * mask_t)
  
  predictions = [data_process.convertToStr(line, data_process.chs_idx_vocab) for line in torch.cat(predictions, dim=1)]

  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()

  if is_predict:
    random_idx = random.sample(range(batch_size), k=sample_k)
    with codecs.open(predict_path, 'a', 'utf-8') as file:
      for idx in random_idx:
        to_write = 'SRC: {}\nTGT: {}\nGOLDEN: {}\n\n'.format(src_sentences[idx], predictions[idx], tgt_sentences[idx])
        file.write(to_write)
        file.flush()
  
  return (loss / torch.sum(tgt_loss_mask)).item()

if __name__ == '__main__':
  data_process = DataProcess('data/cmn-eng/cmn.txt')
  encoder = Encoder(input_size=6707, hidden_size=320)
  decoder = Decoder(hidden_size=320, output_size=3474)
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=5e-3)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=5e-3)
  criterion = nn.NLLLoss(reduction='none')

  train_steps = 1
  all_losses = []
  act_step = 1
  for _ in tqdm(range(train_steps)):
    for datas in data_process.dataBatch(100):
      loss = train(datas, {'encoder': encoder_optimizer, 'decoder': decoder_optimizer},
                   True, 'data/prediction.txt')
      all_losses.append(loss)

      print('Step: {}\t Loss: {:2f}'.format(act_step, loss))
      act_step +=1

  print('Finish Training!')
  torch.save(encoder.state_dict(), 'models/encoder.pth')
  torch.save(decoder.state_dict(), 'models/decoder.pth')

  plt.figure()
  plt.plot(all_losses)
  plt.show()

  # step = 1
  # all_losses = []
  # for step in tqdm(range(train_steps)):
  #   for data_batch in data_process.dataBatch(10):
  #     src_input_data = data_batch['src_input_data']
  #     src_seq_length = data_batch['src_seq_length']
  #     tgt_input_data = data_batch['tgt_input_data']
  #     tgt_output_data = data_batch['tgt_output_data']
  #     tgt_seq_length = data_batch['tgt_seq_length']
  #     batch_size = data_batch['actual_bs']

  #     src_sentences = [data_process.convertToStr(line, data_process.eng_idx_vocab) for line in src_input_data]
  #     tgt_sentences = [data_process.convertToStr(line, data_process.chs_idx_vocab) for line in tgt_output_data]

  #     encoder_optimizer.zero_grad()
  #     decoder_optimizer.zero_grad()

  #     # Encoder
  #     encoder_hidden = encoder.initHidden(batch_size)
  #     encoder_outputs = encoder(src_input_data, src_seq_length, encoder_hidden)

  #     # attention mask
  #     max_length = torch.max(src_seq_length)
  #     attention_mask = torch.arange(max_length)[None, :] < src_seq_length[:, None]
  #     attention_mask = attention_mask.float()[:, None, :]
  #     attention_mask = (attention_mask - 1) * 10000

  #     # Decoder
  #     decoder_hidden = decoder.initHidden(batch_size)
  #     seq_length = tgt_input_data.size()[1]
  #     tgt_loss_mask = torch.arange(torch.max(tgt_seq_length))[None, :] < tgt_seq_length[:, None]
  #     tgt_loss_mask = tgt_loss_mask.float()
      
  #     loss = 0
  #     predictions = []
  #     for t in range(seq_length):
  #       # (b, 1)
  #       input_t = tgt_input_data[:, t][:, None]
  #       golden_t = tgt_output_data[:, t]
  #       mask_t = tgt_loss_mask[:, t]

  #       output, decoder_hidden = decoder(input_t, decoder_hidden, encoder_outputs, attention_mask)
  #       topv, topi = output.topk(1)
  #       predictions.append(topi)
  #       loss += torch.sum(criterion(output, golden_t) * mask_t)
    
  #     predictions = [data_process.convertToStr(line, data_process.chs_idx_vocab) for line in torch.cat(predictions, dim=1)]

  #     loss.backward()
  #     encoder_optimizer.step()
  #     decoder_optimizer.step()
  #     all_losses.append((loss / torch.sum(tgt_loss_mask).item()))
  #     print('Step: {}\t Loss: {:2f}'.format(step, loss / torch.sum(tgt_loss_mask)))
      
  #     predicted_idx = random.sample(range(batch_size), k=3)
  #     with codecs.open('data/cmn-eng/prediciton.txt', 'a', 'utf-8') as file:
  #       for idx in predicted_idx:
  #         to_write = 'SRC: {}\nTGT: {}\nGOLDEN: {}\n\n'.format(src_sentences[idx], predictions[idx], tgt_sentences[idx])
  #         file.write(to_write)
  #         file.flush()
  


  # test_sentence = 'hug tom'
  # input_data = data_process._convertToIdx(data_process._cleanLine(test_sentence), vocab_idx=data_process.eng_vocab_idx, split_tag=' ')
  # seq_length = torch.tensor([len(input_data)])
  # input_data = torch.tensor(input_data)[None, :]

  # encoder.load_state_dict(torch.load('models/encoder.pth'))
  # encoder_hidden = encoder.initHidden(1)
  # encoder_outputs = encoder(input_data, seq_length, encoder_hidden)

  # decoder.load_state_dict(torch.load('models/decoder.pth'))
  # decoder_hidden = decoder.initHidden(1)

  # # attention mask
  # max_length = torch.max(seq_length)
  # attention_mask = torch.arange(max_length)[None, :] < seq_length[:, None]
  # attention_mask = attention_mask.float()[:, None, :]
  # attention_mask = (attention_mask - 1) * 10000

  # predict_t = torch.tensor([data_process.chs_vocab_idx[SOS]])[None, :]
  # results = []
  # while predict_t.item() != data_process.chs_vocab_idx[EOS]:
  #   output, decoder_hidden = decoder(predict_t, decoder_hidden, encoder_outputs, attention_mask)
  #   topv, topi = output.topk(1)
  #   predict_t = topi
  #   results.append(predict_t.item())

  # print(data_process.convertToStr(results, data_process.chs_idx_vocab))