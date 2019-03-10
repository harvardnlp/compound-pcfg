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
import torch.nn as nn
import numpy as np
import time
from utils import *
import re
import pickle

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--data_file', default='')
parser.add_argument('--model_file', default='')
parser.add_argument('--out_file', default='pred-parse.txt')
# Inference options
parser.add_argument('--use_mean', default=1, type=int, help='use mean from q if = 1')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')

def clean_number(w):    
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w
  
def main(args):
  print('loading model from ' + args.model_file)
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  cuda.set_device(args.gpu)
  model.eval()
  model.cuda()
  word2idx = checkpoint['word2idx']
  pred_out = open(args.out_file, "w")
  with torch.no_grad():
    for sent_orig in open(args.data_file, "r"):
      # punctuation should be removed
      sent = sent_orig.lower().strip().split()
      sent = [clean_number(w) for w in sent]
      sent_orig = sent_orig.strip().split()
      length = len(sent)
      if length == 1:
        continue 
      sent_idx = [word2idx[w] if w in word2idx else word2idx["<unk>"] for w in sent]
      sents = torch.from_numpy(np.array(sent_idx)).unsqueeze(0)
      sents = sents.cuda()
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=(args.use_mean==1))
      pred_span= [(a[0], a[1]) for a in argmax_spans[0]]
      argmax_tags = model.tags[0]
      binary_matrix = binary_matrix[0].cpu().numpy()
      label_matrix = np.zeros((length, length))
      for span in argmax_spans[0]:
        label_matrix[span[0]][span[1]] = span[2]
      pred_tree = {}
      for i in range(length):
        tag = "T-" + str(int(argmax_tags[i].item())+1) 
        pred_tree[i] = "(" + tag + " " + sent_orig[i] + ")"
      for k in np.arange(1, length):
        for s in np.arange(length):
          t = s + k
          if t > length - 1: break
          if binary_matrix[s][t] == 1:
            nt = "NT-" + str(int(label_matrix[s][t])+1)
            span = "(" + nt + " " + pred_tree[s] + " " + pred_tree[t] +  ")"
            pred_tree[s] = span
            pred_tree[t] = span
      pred_tree = pred_tree[0]
      pred_out.write(pred_tree.strip() + "\n")
      print(pred_tree)
  pred_out.close()

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
