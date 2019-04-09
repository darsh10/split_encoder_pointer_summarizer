from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random
import sys

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.encoder_division = nn.Linear(config.hidden_dim * 4, 1)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)
	    self.p_gen_encoder = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs_list, encoder_feature_list, enc_padding_mask_list,
                c_t_1_list, extra_zeros_list, enc_batch_extend_vocab_list, coverage_list, step):

        c_t_list = []
        attn_dist_list = []
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                      c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            for ind in range(len(coverage_list)):
                c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs_list[ind], encoder_feature_list[ind], enc_padding_mask_list[ind], coverage_list[ind])
                c_t_list.append(c_t)
                coverage_list[ind] = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        c_t_1 = None
        p_dec = F.sigmoid(self.encoder_division(torch.cat((c_t_1_list[0], c_t_1_list[1]),-1)))
        if len(sys.argv) > 2:
	    p_dec = (p_dec + int(sys.argv[2]))/101
        for ind,value in enumerate(c_t_1_list):
            if c_t_1 is None:
                #c_t_1 = c_t_1_list[ind]
		c_t_1 = (p_dec) * c_t_1_list[ind]
            else:
                #c_t_1 += c_t_1_list[ind]
		c_t_1 += (1-p_dec) *c_t_1_list[ind]
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t_list = []
        for ind in range(len(coverage_list)):
            c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs_list[ind], encoder_feature_list[ind], enc_padding_mask_list[ind], coverage_list[ind])
            c_t_list.append(c_t)
            attn_dist_list.append(attn_dist)
            if self.training or step > 0:
                coverage_list[ind] = coverage_next

        p_gen = None
	p_enc = None
        if config.pointer_gen:
            c_t = None
            assert len(c_t_list) > 0
            for ind in range(len(c_t_list)):
                if c_t is None:
                    c_t = c_t_list[ind]
                else:
                    c_t += c_t_list[ind]
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

	    p_enc = self.p_gen_encoder(p_gen_input)
            p_enc = F.sigmoid(p_enc)
	    if len(sys.argv) > 2:
                p_enc = (p_enc + int(sys.argv[2]))/101

        c_t = None
        for ind in range(len(c_t_list)):
            if c_t is None:
                c_t = c_t_list[ind] #p_enc * c_t_list[ind]
            else:
                c_t = c_t + c_t_list[ind] #c_t + (1-p_enc) * c_t_list[ind]
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)


        attn_dist_list_ = []
        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            for ind,attn_dist in enumerate(attn_dist_list):
		if ind == 0:
                    attn_dist_list_.append(p_enc * (1 - p_gen) * attn_dist)
		else:
		    attn_dist_list_.append((1 - p_enc) * (1 - p_gen) * attn_dist)

            for extra_zeros in extra_zeros_list:
                if extra_zeros is not None:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            for enc_batch_extend_vocab,attn_dist_ in zip(enc_batch_extend_vocab_list, attn_dist_list_):
                vocab_dist_ = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
            final_dist = vocab_dist_
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t_list, attn_dist_list, p_gen, coverage_list

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
