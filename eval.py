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

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--data_file', default='data/ptb-test.txt')
parser.add_argument('--model_file', default='')
parser.add_argument('--out_file', default='pred-parse.txt')
# Inference options
parser.add_argument('--use_mean', default=1, type=int, help='use mean from q if = 1')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions

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
  total_kl = 0.
  total_nll = 0.
  num_sents = 0
  num_words = 0
  word2idx = checkpoint['word2idx']
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  pred_out = open(args.out_file, "w")
  with torch.no_grad():
    for tree in open(args.data_file, "r"):
      tree = tree.strip()
      action = get_actions(tree)
      tags, sent, sent_lower = get_tags_tokens_lowercase(tree)
      gold_span, binary_actions, nonbinary_actions = get_nonbinary_spans(action)
      length = len(sent)
      sent_orig = sent_lower
      sent = [clean_number(w) for w in sent_orig]
      if length == 1:
        continue # we ignore length 1 sents. this doesn't change F1 since we discard trivial spans
      sent_idx = [word2idx[w] if w in word2idx else word2idx["<unk>"] for w in sent]
      sents = torch.from_numpy(np.array(sent_idx)).unsqueeze(0)
      sents = sents.cuda()
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=(args.use_mean==1))
      total_nll += nll.sum().item()
      total_kl  += kl.sum().item()
      num_sents += 1
      # the grammar implicitly generates </s> token, in contrast to a sequential lm which must explicitly
      # generate it. the sequential lm takes into account </s> token in perplexity calculations, so
      # for comparison the pcfg must also take into account </s> token, which amounts to just adding
      # one more token to length for each sentence
      num_words += length + 1
      pred_span= [(a[0], a[1]) for a in argmax_spans[0]]
      pred_span_set = set(pred_span[:-1]) #the last span in the list is always the
      gold_span_set = set(gold_span[:-1]) #trival sent-level span so we ignore it
      tp, fp, fn = get_stats(pred_span_set, gold_span_set) 
      corpus_f1[0] += tp
      corpus_f1[1] += fp
      corpus_f1[2] += fn
      # sent-level F1 is based on code from https://github.com/yikangshen/PRPN
      overlap = pred_span_set.intersection(gold_span_set)
      prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
      reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
      if len(pred_span_set) == 0 or len(gold_span_set) == 0:
        continue
      f1 = 2 * prec * reca / (prec + reca + 1e-8)
      sent_f1.append(f1)
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
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words)
  kl = total_kl /num_sents
  # note that if use_mean == 1, then the PPL upper bound is not a true upper bound
  # run with use_mean == 0, to get the true upper bound
  print('ReconPPL: %.2f, KL: %.4f, PPL Upper Bound from ELBO: %.2f' %
        (recon_ppl, kl, ppl_elbo))
  print('Corpus F1: %.2f, Sentence F1: %.2f' %
        (corpus_f1*100, np.mean(np.array(sent_f1))*100))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
