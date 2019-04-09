from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad, Adam

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adam(params, lr=initial_lr)#Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch_list, enc_padding_mask_list, enc_lens_list, enc_batch_extend_vocab_list, extra_zeros_list, c_t_1_list, coverage_list = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

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
            sorted_encoder_outputs, sorted_encoder_feature, sorted_encoder_hidden = self.model.encoder(sorted_enc_batch, sorted_enc_lens)
            encoder_outputs = torch.index_select(sorted_encoder_outputs, 0, torch.LongTensor(inverse_sorted_indices) if not use_cuda else torch.LongTensor(inverse_sorted_indices).cuda())
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

        #c_t_1_list = [c_t_1]
        #coverage_list = [coverage]

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1_list, attn_dist_list, p_gen, next_coverage_list = self.model.decoder(y_t_1, s_t_1, encoder_outputs_list, encoder_feature_list, enc_padding_mask_list, c_t_1_list, extra_zeros_list, enc_batch_extend_vocab_list, coverage_list, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = 0.0
                for ind in range(len(coverage_list)):
                    step_coverage_loss += torch.sum(torch.min(attn_dist_list[ind], coverage_list[ind]), 1)
                    coverage_list[ind] = next_coverage_list[ind]
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 500
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 500 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
