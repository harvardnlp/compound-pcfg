#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from models import CompPCFG
from torch.nn.init import xavier_uniform_

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='../neural-pcfg/data_files/ptb-10k-batch16-train.pkl')
parser.add_argument('--val_file', default='')
parser.add_argument('--model_file', default='../neural-pcfg/compound-pcfg-convert.pt')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=17, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)    
  model = CompPCFG(vocab = vocab_size,
                   state_dim = args.state_dim,
                   t_states = args.t_states,
                   nt_states = args.nt_states,
                   h_dim = args.h_dim,
                   w_dim = args.w_dim,
                   z_dim = args.z_dim)
  print('loading model from ' + args.model_file)
  model2_param = torch.load(args.model_file)
  for name, param in model.named_parameters():    
    print(name)
    param2 = model2_param[name]
    param.data.copy_(param2)
  checkpoint = {
    'args': args.__dict__,
    'model': model.cpu(),
    'word2idx': train_data.word2idx,
    'idx2word': train_data.idx2word
  }
  print('Saving checkpoint to %s' % args.save_path)
  torch.save(checkpoint, args.save_path)
  assert False

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
