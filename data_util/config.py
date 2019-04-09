
import os

root_dir = os.path.expanduser("~")

train_data_path = "/scratch1/darsh/cnn-dailymail/finished_files/chunked/train_*"
eval_data_path = "/scratch1/darsh/cnn-dailymail/finished_files/val.bin"
decode_data_path = "/scratch1/darsh/cnn-dailymail/finished_files/test.bin"
vocab_path = "/scratch1/darsh/cnn-dailymail/finished_files/vocab"
#train_data_path = "/data/rsg/nlp/tals/fakenews/datasets/summarization/finished_files/chunked/train_*"
#eval_data_path = "/data/rsg/nlp/tals/fakenews/datasets/summarization/finished_files/val.bin"
#decode_data_path = "/data/rsg/nlp/tals/fakenews/datasets/summarization/finished_files/test.bin"
#vocab_path = "/data/rsg/nlp/tals/fakenews/datasets/summarization/finished_files/vocab"
log_root = "/scratch1/darsh/pointer_summarizer"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 64
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.001#0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 50000/8

use_gpu=True

lr_coverage=0.15
