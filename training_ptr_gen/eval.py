from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model
import sys
reload(sys)
sys.setdefaultencoding('utf8')

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = tf.summary.FileWriter(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch_list, enc_padding_mask_list, enc_lens_list, enc_batch_extend_vocab_list, extra_zeros_list, c_t_1_list, coverage_list = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

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
            sorted_enc_batch = torch.index_select(enc_batch, 0, torch.LongTensor(sorted_indices) if not use_cuda else torch.LongTensor(sorted_indices).cuda())
            sorted_enc_lens = enc_lens[sorted_indices]
            sorted_encoder_outputs, sorted_encoder_feature, sorted_encoder_hidden = self.model.encoder(sorted_enc_batch
, sorted_enc_lens)
            encoder_outputs = torch.index_select(sorted_encoder_outputs, 0, torch.LongTensor(inverse_sorted_indices) if
 not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda())
            encoder_feature = torch.index_select(sorted_encoder_feature.view(encoder_outputs.shape), 0, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda()).view(sorted_encoder_feature.shape)
            encoder_hidden = tuple([torch.index_select(sorted_encoder_hidden[0], 1, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda()), torch.index_select(sorted_encoder_hidden[1], 1, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda())])
            #encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
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
        #s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        target_words = []
        output_words = []
        id_to_words = {v: k for k, v in self.vocab.word_to_id.iteritems()}
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1_list,attn_dist_list, p_gen, next_coverage_list = self.model.decoder(y_t_1, s_t_1, encoder_outputs_list, encoder_feature_list, enc_padding_mask_list, c_t_1_list, extra_zeros_list, enc_batch_extend_vocab_list, coverage_list, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            output_ids = final_dist.max(1)[1]
            output_2_candidates = final_dist.topk(2,1)[1]
            for ind in range(output_ids.shape[0]):
                if self.vocab.word_to_id['X'] == output_ids[ind].item():
                    output_ids[ind] = output_2_candidates[ind][1]
            target_step = []
            output_step = []
            step_mask = dec_padding_mask[:, di] 
            for i in range(target.shape[0]):
                if target[i].item() >= len(id_to_words) or step_mask[i].item() == 0:
                    target[i] = 0
                target_step.append(id_to_words[target[i].item()])
                if output_ids[i].item() >= len(id_to_words) or step_mask[i].item() == 0:
                    output_ids[i] = 0
                output_step.append(id_to_words[output_ids[i].item()])
            target_words.append(target_step)
            output_words.append(output_step)
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                #step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                #step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                #coverage = next_coverage
                step_coverage_loss = 0.0
                for ind in range(len(coverage_list)):
                    step_coverage_loss += torch.sum(torch.min(attn_dist_list[ind], coverage_list[ind]), 1)
                    coverage_list[ind] = next_coverage_list[ind]
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss 

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        self.write_words(output_words,"output.txt")
        self.write_words(target_words,"input.txt")

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.item()

    def write_words(self, len_batch_words, output_file):
        batch_sentences = ["" for _ in range(len(len_batch_words[0]))]
        batch_sentences_done = [False for _ in range(len(len_batch_words[0]))]
        for i in range(len(len_batch_words)):
            for j in range(len(len_batch_words[i])):
                if len_batch_words[i][j] == "[STOP]":
                    batch_sentences_done[j] = True
                if batch_sentences_done[j] != True:
                    batch_sentences[j] += len_batch_words[i][j] + " "
        f = open(output_file, "a")
        for sentence in batch_sentences:
            f.write(sentence.strip() + "\n")
        f.close()


    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()


if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()


