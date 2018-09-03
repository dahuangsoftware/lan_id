#!/usr/bin/env python
#coding=utf-8
#基于文档频率选择特征

MAX_NGRAM_ORDER = 4 # 最大的gram
TOKENS_PER_ORDER = 15000 # 每一个序列考虑的令牌数

import os, sys, argparse
import collections
import csv
import shutil
import tempfile
import marshal
import random
import numpy
import cPickle
import multiprocessing as mp
import atexit
import gzip
from itertools import tee, imap, islice
from collections import defaultdict
from datetime import datetime
from contextlib import closing

from common import Enumerator, unmarshal_iter, MapPool, write_features, write_weights

#计算文档频率
def pass_sum_df(bucket):
  doc_count = defaultdict(int)
  count = 0
  with gzip.open(os.path.join(bucket, "docfreq"),'wb') as docfreq:
    for path in os.listdir(bucket):
      if path.endswith('.domain'):
        for key, _, value in unmarshal_iter(os.path.join(bucket,path)):
          doc_count[key] += value
          count += 1
    
    for item in doc_count.iteritems():
      docfreq.write(marshal.dumps(item))
  return count

#计算每一个特征的个数
def tally(bucketlist, jobs=None):
  with MapPool(jobs) as f:
    pass_sum_df_out = f(pass_sum_df, bucketlist)

    for i, keycount in enumerate(pass_sum_df_out):
      print "processed bucket (%d/%d) [%d keys]" % (i+1, len(bucketlist), keycount)

  # 构造feature到频率的映射
  doc_count = {}
  for bucket in bucketlist:
    for key, value in unmarshal_iter(os.path.join(bucket, 'docfreq')):
      doc_count[key] = value

  return doc_count

#DF特征的选择
def ngram_select(doc_count, max_order=MAX_NGRAM_ORDER, tokens_per_order=TOKENS_PER_ORDER):
  features = set()
  for i in range(1, max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    features |= set(sorted(d, key=d.get, reverse=True)[:tokens_per_order])
  features = sorted(features)
  
  return features