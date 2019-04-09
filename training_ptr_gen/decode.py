#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            output_ids = [x if x<len(self.vocab.word_to_id) else 0 for x in output_ids]
            if len(batch.art_oovs) == 0:
                batch.art_oovs = [None]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]
            print(original_abstract_sents, decoded_words)

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch_list, enc_padding_mask_list, enc_lens_list, enc_batch_extend_vocab_list, extra_zeros_list, c_t_0_list, coverage_t_0_list = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs_list = []
        encoder_feature_list = []
        s_t_1 = None
        s_t_1_0 = None
        s_t_1_1 = None
        for enc_batch,enc_lens in zip(enc_batch_list, enc_lens_list):
            sorted_indices = sorted(range(len(enc_lens)),key=enc_lens.__getitem__)
            sorted_indices.reverse()
            inverse_sorted_indices = [-1 for _ in range(len(sorted_indices))]
            for index,position in enumerate(sorted_indices):
                inverse_sorted_indices[position] = index
            sorted_enc_batch = torch.index_select(enc_batch, 0, torch.LongTensor(sorted_indices) if not use_cuda       else torch.LongTensor(sorted_indices).cuda())
            sorted_enc_lens = enc_lens[sorted_indices]
            sorted_encoder_outputs, sorted_encoder_feature, sorted_encoder_hidden = self.model.encoder(sorted_enc_batch, sorted_enc_lens)
            encoder_outputs = torch.index_select(sorted_encoder_outputs, 0, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda())
            encoder_feature = torch.index_select(sorted_encoder_feature.view(encoder_outputs.shape), 0, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda()).view(sorted_encoder_feature.shape)
            encoder_hidden = tuple([torch.index_select(sorted_encoder_hidden[0], 1, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda()), torch.index_select(sorted_encoder_hidden[1], 1, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda())])
            encoder_outputs_list.append(encoder_outputs)
            encoder_feature_list.append(encoder_feature)
            if s_t_1 is None:
                s_t_1 = self.model.reduce_state(encoder_hidden)
                s_t_1_0, s_t_1_1 = s_t_1
            else:
                s_t_1_new = self.model.reduce_state(encoder_hidden)
                s_t_1_0 = s_t_1_0 + s_t_1_new[0]
                s_t_1_1 = s_t_1_1 + s_t_1_new[1]
            s_t_1 = tuple([s_t_1_0, s_t_1_1])


        #encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        #s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_1 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        c_t_0 = c_t_0_list[0] + c_t_0_list[1]

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      #coverage=(coverage_t_0[0] if config.is_coverage else None))
                      coverage=None)
                 for _ in xrange(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            c_t_1_list = [c_t_1,c_t_1]
            coverage_t_1_list = [None,None]
            final_dist, s_t, c_t_list, attn_dist_list, p_gen, coverage_t_list = self.model.decoder(y_t_1, s_t_1, encoder_outputs_list, encoder_feature_list, enc_padding_mask_list, c_t_1_list, extra_zeros_list, enc_batch_extend_vocab_list, coverage_t_1_list, steps)

            topk_log_probs, topk_ids = torch.topk(final_dist, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in xrange(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t_list[0][i] + c_t_list[1][i]
                coverage_i = None#(coverage_t[i] if config.is_coverage else None)

                for j in xrange(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()


