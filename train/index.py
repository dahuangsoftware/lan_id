#!/usr/bin/env python
#coding=utf-8

TRAIN_PROP = 1.0
MIN_DOMAIN = 1 # 一种语言最少包含的种类数是1

import os, sys, argparse
import csv
import random
import numpy
from itertools import tee, imap, islice
from collections import defaultdict
from common import Enumerator, makedir

#用于索引语料库中的内容
class CorpusIndexer(object):
  def __init__(self, root, min_domain=MIN_DOMAIN, proportion=TRAIN_PROP, langs=None, domains=None, line_level=False):
    self.root = root
    self.min_domain = min_domain
    self.proportion = proportion 

    if langs is None:
      self.lang_index = defaultdict(Enumerator())
    else:
      # 预先制定了语料集合
      self.lang_index = dict((k,v) for v,k in enumerate(langs))

    if domains is None:
      self.domain_index = defaultdict(Enumerator())
    else:
      # 预先定义了domain集合
      self.domain_index = dict((k,v) for v,k in enumerate(domains))

    self.coverage_index = defaultdict(set)
    self.items = list()

    if os.path.isdir(root):
      candidates = []
      for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for docname in filenames:
          candidates.append(os.path.join(dirpath, docname))
    else:
      candidates = map(str.strip, open(root))

    if line_level:
      self.index_line(candidates)
    else:
      self.index(candidates)

    self.prune_min_domain(self.min_domain)

  def index_line(self, candidates):
    if self.proportion < 1.0:
      raise NotImplementedError("proportion selection not available for file-per-class")

    for path in candidates:
      d, lang = os.path.split(path)
      d, domain = os.path.split(d)

      # 建索引
      try:
        domain_id = self.domain_index[domain]
        lang_id = self.lang_index[lang]
      except KeyError:
        #超过了范围
        continue

      # 将domain-lang值加入
      self.coverage_index[domain].add(lang)

      with open(path) as f:
        for i,row in enumerate(f):
          docname = "line{0}".format(i)
          self.items.append((domain_id,lang_id,docname,path))

  def index(self, candidates):

    #建路径
    for path in candidates:
      if random.random() < self.proportion:

        d, docname = os.path.split(path)
        d, lang = os.path.split(d)
        d, domain = os.path.split(d)

        try:
          domain_id = self.domain_index[domain]
          lang_id = self.lang_index[lang]
        except KeyError:
          continue
        self.coverage_index[domain].add(lang)
        self.items.append((domain_id,lang_id,docname,path))

  def prune_min_domain(self, min_domain):
    lang_domain_count = defaultdict(int)
    for langs in self.coverage_index.values():
      for lang in langs:
        lang_domain_count[lang] += 1
    reject_langs = set( l for l in lang_domain_count if lang_domain_count[l] < min_domain)

    # 从index中移除语言
    if reject_langs:
      reject_ids = set(self.lang_index[l] for l in reject_langs)
      new_lang_index = defaultdict(Enumerator())
      lm = dict()
      for k,v in self.lang_index.items():
        if v not in reject_ids:
          new_id = new_lang_index[k]
          lm[v] = new_id
      # 消除语言的所有条目
      self.items = [ (d, lm[l], n, p) for (d, l, n, p) in self.items if l in lm]

      self.lang_index = new_lang_index

  @property
  def dist_lang(self):
    retval = numpy.zeros((len(self.lang_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[l] += 1
    return retval

  # 基于域的向量值
  @property
  def dist_domain(self):
    retval = numpy.zeros((len(self.domain_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[d] += 1
    return retval
