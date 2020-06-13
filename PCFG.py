#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import random
  
class PCFG(nn.Module):
  def __init__(self, nt_states, t_states):
    super(PCFG, self).__init__()
    self.nt_states = nt_states
    self.t_states = t_states
    self.states = nt_states + t_states
    self.huge = 1e9

  def logadd(self, x, y):
    d = torch.max(x,y)  
    return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    

  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d

  def _inside(self, unary_scores, rule_scores, root_scores):
    #inside step
    #unary scores : b x n x T
    #rule scores : b x NT  x (NT+T) x (NT+T)
    #root : b x NT
    batch_size = unary_scores.size(0)
    n = unary_scores.size(1)
    self.beta = unary_scores.new(batch_size, n, n, self.states).fill_(-self.huge)
    for k in range(n):
      for state in range(self.t_states):
        self.beta[:, k, k, self.nt_states + state] = unary_scores[:, k, state]        
    for w in np.arange(1, n+1):
      for s in range(n):
        t = s + w
        if t > n-1:
          break
        tmp_u = []
        for u in np.arange(s, t):
          if s == u:
            l_start = self.nt_states
            l_end = self.states
          else:
            l_start = 0
            l_end = self.nt_states
          if u+1 == t:
            r_start = self.nt_states
            r_end = self.states
          else: 
            r_start = 0
            r_end = self.nt_states
          tmp_rule_scores = rule_scores[:, :, l_start:l_end, r_start:r_end] # b x NT x NT+T X NT+T
          beta_left = self.beta[:, s, u, l_start:l_end] #  b x NT
          beta_right = self.beta[:, u+1, t, r_start:r_end] # b x NT
          beta_left = beta_left.unsqueeze(2).unsqueeze(1)
          beta_right = beta_right.unsqueeze(1).unsqueeze(2)
          tmp_scores = beta_left + beta_right + tmp_rule_scores # b x NT x NT+T x NT+T
          tmp_scores = tmp_scores.view(batch_size, self.nt_states, -1)
          tmp_u.append(self.logsumexp(tmp_scores, 2).unsqueeze(2))
        tmp_u = torch.cat(tmp_u, 2)
        tmp_u = self.logsumexp(tmp_u, 2)
        self.beta[:, s, t, :self.nt_states] = tmp_u[:, :self.nt_states]
    log_Z = self.beta[:, 0, n-1, :self.nt_states] + root_scores
    log_Z = self.logsumexp(log_Z, 1)
    return log_Z

  def _viterbi(self, unary_scores, rule_scores, root_scores):
    #unary scores : b x n x T
    #rule scores : b x NT x (NT+T) x (NT+T)
    batch_size = unary_scores.size(0)
    n = unary_scores.size(1)
    self.scores = unary_scores.new(batch_size, n, n, self.states).fill_(-self.huge)
    self.bp = unary_scores.new(batch_size, n, n, self.states).fill_(-1)
    self.left_bp = unary_scores.new(batch_size, n, n, self.states).fill_(-1)
    self.right_bp = unary_scores.new(batch_size, n, n, self.states).fill_(-1)
    self.argmax = unary_scores.new(batch_size, n, n).fill_(-1)
    self.argmax_tags = unary_scores.new(batch_size, n).fill_(-1)
    self.spans = [[] for _ in range(batch_size)]
    for k in range(n):
      for state in range(self.t_states):
        self.scores[:, k, k, self.nt_states + state] = unary_scores[:, k, state]        
    for w in np.arange(1, n+1):
      for s in range(n):
        t = s + w
        if t > n-1:
          break
        tmp_max_score = []
        tmp_left_child = []
        tmp_right_child = []
        for u in np.arange(s, t):
          if s == u:
            l_start = self.nt_states
            l_end = self.states
          else:
            l_start = 0
            l_end = self.nt_states
          if u+1 == t:
            r_start = self.nt_states
            r_end = self.states
          else: 
            r_start = 0
            r_end = self.nt_states
          tmp_rule_scores = rule_scores[:, :, l_start:l_end, r_start:r_end] # b x NT x NT+T X NT+T
          beta_left = self.scores[:, s, u, l_start:l_end] #  b x NT
          beta_right = self.scores[:, u+1, t, r_start:r_end] # b x NT
          beta_left = beta_left.unsqueeze(2).unsqueeze(1)
          beta_right = beta_right.unsqueeze(1).unsqueeze(2)
          tmp_scores = beta_left + beta_right + tmp_rule_scores # b x NT x NT+T x NT+T
          r_states = tmp_scores.size(3)
          tmp_scores_flat = tmp_scores.view(batch_size, tmp_scores.size(1), -1)
          max_score, max_idx = torch.max(tmp_scores_flat, 2) # b x NT
          tmp_max_score.append(max_score.unsqueeze(2))
          left_child = (max_idx.float() / r_states).floor().long()
          right_child = torch.remainder(max_idx, r_states)
          tmp_left_child.append(left_child.unsqueeze(2) + l_start)
          tmp_right_child.append(right_child.unsqueeze(2) + r_start)
        tmp_max_score = torch.cat(tmp_max_score, 2) # b x NT x u
        tmp_left_child = torch.cat(tmp_left_child, 2)
        tmp_right_child = torch.cat(tmp_right_child, 2)
        max_score, max_idx = torch.max(tmp_max_score, 2)
        max_left_child = torch.gather(tmp_left_child, 2, max_idx.unsqueeze(2)).squeeze(2)
        max_right_child = torch.gather(tmp_right_child, 2, max_idx.unsqueeze(2)).squeeze(2)
        self.scores[:, s, t, :self.nt_states] = max_score[:, :self.nt_states]
        self.bp[:, s, t, :self.nt_states] = max_idx[:, :self.nt_states] + s          
        self.left_bp[:, s, t, :self.nt_states] = max_left_child[:, :self.nt_states]
        self.right_bp[:, s, t, :self.nt_states] = max_right_child[:, :self.nt_states]
    max_score = self.scores[:, 0, n-1, :self.nt_states] + root_scores
    max_score, max_idx = torch.max(max_score, 1)
    for b in range(batch_size):
      self._backtrack(b, 0, n-1, max_idx[b].item())      
    return self.scores[:, 0, n-1, 0], self.argmax, self.spans

  def _backtrack(self, b, s, t, state):
    u = int(self.bp[b][s][t][state])
    assert(s <= t)
    left_state = int(self.left_bp[b][s][t][state])
    right_state = int(self.right_bp[b][s][t][state])
    self.argmax[b][s][t] = 1
    if s == t:
      self.argmax_tags[b][s] = state - self.nt_states
      return None      
    else:
      self.spans[b].insert(0, (s,t, state))
      self._backtrack(b, s, u, left_state)
      self._backtrack(b, u+1, t, right_state)
    return None  
