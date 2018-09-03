#!/usr/bin/env python
#coding=utf-8
#特征提取
FEATURES_PER_LANG = 300 #  每一种语言选取特征的数量

import os, sys, argparse
import csv
import marshal
import numpy
import multiprocessing as mp
from collections import defaultdict
from common import read_weights, Enumerator, write_features

#选取LD特征
#参数ignore_domain表明是否使用域权重
def select_LD_features(ig_lang, ig_domain, feats_per_lang, ignore_domain=False):
  assert (ig_domain is None) or (len(ig_lang) == len(ig_domain))
  num_lang = len(ig_lang.values()[0])
  num_term = len(ig_lang)

  term_index = defaultdict(Enumerator())


  ld = numpy.empty((num_lang, num_term), dtype=float)

  for term in ig_lang:
    term_id = term_index[term]
    if ignore_domain:
      ld[:, term_id] = ig_lang[term]
    else:
      ld[:, term_id] = ig_lang[term] - ig_domain[term]

  terms = sorted(term_index, key=term_index.get)
  selected_features = dict()
  for lang_id, lang_w in enumerate(ld):
    term_inds = numpy.argsort(lang_w)[-feats_per_lang:]
    selected_features[lang_id] = [terms[t] for t in term_inds]

  return selected_features