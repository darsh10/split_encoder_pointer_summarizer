from torch.autograd import Variable
import numpy as np
import torch
from data_util import config

def get_input_from_batch(batch, use_cuda):
  batch_size = len(batch.enc_lens)

  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens

  enc_batch_2 = Variable(torch.from_numpy(batch.enc_batch_2).long())
  enc_padding_mask_2 = Variable(torch.from_numpy(batch.enc_padding_mask_2).float())
  enc_lens_2 = batch.enc_lens_2

  extra_zeros = None
  enc_batch_extend_vocab = None

  extra_zeros_2 = None
  enc_batch_extend_vocab_2 = None 

  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    enc_batch_extend_vocab_2 = Variable(torch.from_numpy(batch.enc_batch_extend_vocab_2).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0 or batch.max_art_oovs_2 > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
      extra_zeros_2 = Variable(torch.zeros((batch_size, batch.max_art_oovs_2)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
  c_t_1_2 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim))) 

  coverage = None
  coverage_2 = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))
    coverage_2 = Variable(torch.zeros(enc_batch_2.size()))

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()
    enc_batch_2 = enc_batch_2.cuda()
    enc_padding_mask_2 = enc_padding_mask_2.cuda()

    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
      enc_batch_extend_vocab_2 = enc_batch_extend_vocab_2.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
      extra_zeros_2 = extra_zeros_2.cuda()
    c_t_1 = c_t_1.cuda()
    c_t_1_2 = c_t_1_2.cuda()

    if coverage is not None:
      coverage = coverage.cuda()
      coverage_2 = coverage_2.cuda()

  return [enc_batch, enc_batch_2], [enc_padding_mask, enc_padding_mask_2], [enc_lens, enc_lens_2], [enc_batch_extend_vocab, enc_batch_extend_vocab_2], [extra_zeros, extra_zeros_2], [c_t_1, c_t_1_2], [coverage, coverage_2]

def get_output_from_batch(batch, use_cuda):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

