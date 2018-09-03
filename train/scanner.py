#!/usr/bin/env python
#coding=utf-8
#根据特性集构建扫描仪

import cPickle
import os, sys, argparse 
import array
from collections import deque, defaultdict
from common import read_features

class Scanner(object):
  alphabet = map(chr, range(1<<8))
  @classmethod
  def from_file(cls, path):
    with open(path) as f:
      tk_nextmove, tk_output, feats = cPickle.load(f)
    if isinstance(feats, dict):
      raise ValueError("old format scanner - please retrain. see code for details.")
    tk_output_f = dict( (k,[feats[i] for i in v]) for k,v in tk_output.iteritems() )
    scanner = cls.__new__(cls)
    scanner.__setstate__((tk_nextmove, tk_output_f))
    return scanner
    
  def __init__(self, keywords):
    self.build(keywords)

  def __call__(self, value):
    return self.search(value)

  def build(self, keywords):
    goto = dict()
    fail = dict()
    output = defaultdict(set)

    # 算法2
    newstate = 0
    for a in keywords:
      state = 0
      j = 0
      while (j < len(a)) and (state, a[j]) in goto:
        state = goto[(state, a[j])]
        j += 1
      for p in range(j, len(a)):
        newstate += 1
        goto[(state, a[p])] = newstate
        state = newstate
      output[state].add(a)
    for a in self.alphabet:
      if (0,a) not in goto: 
        goto[(0,a)] = 0

    # 算法3
    queue = deque()
    for a in self.alphabet:
      if goto[(0,a)] != 0:
        s = goto[(0,a)]
        queue.append(s)
        fail[s] = 0
    while queue:
      r = queue.popleft()
      for a in self.alphabet:
        if (r,a) in goto:
          s = goto[(r,a)]
          queue.append(s)
          state = fail[r]
          while (state,a) not in goto:
            state = fail[state]
          fail[s] = goto[(state,a)]
          if output[fail[s]]:
            output[s].update(output[fail[s]])

    # 算法4
    self.nextmove = {}
    for a in self.alphabet:
      self.nextmove[(0,a)] = goto[(0,a)]
      if goto[(0,a)] != 0:
        queue.append(goto[(0,a)])
    while queue:
      r = queue.popleft()
      for a in self.alphabet:
        if (r,a) in goto:
          s = goto[(r,a)]
          queue.append(s)
          self.nextmove[(r,a)] = s
        else:
          self.nextmove[(r,a)] = self.nextmove[(fail[r],a)]

    #输出至元组
    self.output = dict((k, tuple(output[k])) for k in output)

    def generate_nm_arr(typecode):
      def nextstate_iter():
        for state in xrange(newstate+1):
          for letter in self.alphabet:
            yield self.nextmove[(state, letter)]
      return array.array(typecode, nextstate_iter())
    try:
      self.nm_arr = generate_nm_arr('H')
    except OverflowError:
      self.nm_arr = generate_nm_arr('L')

  #编译和输出
  def __getstate__(self):
    return (self.nm_arr, self.output)

  def __setstate__(self, value):
    nm_array, output = value
    self.nm_arr = nm_array
    self.output = output
    self.nextmove = {}
    for i, next_state in enumerate(nm_array):
      state = i / 256
      letter = chr(i % 256)
      self.nextmove[(state, letter)] = next_state 

  def search(self, string):
    state = 0
    for letter in map(ord,string):
      state = self.nm_arr[(state << 8) + letter]
      for key in self.output.get(state, []):
        yield key

#创建扫描仪
def build_scanner(features):
  feat_index = index(features)
  # 构造
  print "building scanner"
  scanner = Scanner(features)
  tk_nextmove, raw_output = scanner.__getstate__()
  tk_output = {}
  for k,v in raw_output.items():
    tk_output[k] = tuple(feat_index[f] for f in v)
  return tk_nextmove, tk_output

#返回一个索引的字典
def index(seq):
  return dict((k,v) for (v,k) in enumerate(seq))
