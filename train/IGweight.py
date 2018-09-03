#!/usr/bin/env python
#coding=utf-8
#给定一个分词好的数据桶以及一个特征集，计算IG权重

import os, sys, argparse 
import csv
import numpy
import multiprocessing as mp
from itertools import tee, imap, islice
from collections import defaultdict
from contextlib import closing
from common import unmarshal_iter, MapPool, Enumerator, write_weights, read_features 

#优化熵的实现，尤其是在长向量方面速度更快，用0来替换可能出现的log(0)
def entropy(v, axis=0):
  v = numpy.array(v, dtype='float')
  s = numpy.sum(v, axis=axis)
  with numpy.errstate(divide='ignore', invalid='ignore'):
    log = numpy.nan_to_num(numpy.log(v))
    rhs = numpy.nansum(v * log, axis=axis) / s
    r = numpy.log(s) - rhs
  return numpy.nan_to_num(r)

#输入features:用于计算IG的特征集，dist:背后的分布，binarize:二值化，suffix:要处理的文件的后缀
def setup_pass_IG(features, dist, binarize, suffix):
  global __features, __dist, __binarize, __suffix
  __features = features
  __dist = dist
  __binarize = binarize
  __suffix = suffix

#计算每一个特征的信息增益
#参数buckets一个目录，含有适当后缀的文件，文件中为（term，event_id,count）三元组
def pass_IG(buckets):
  global __features, __dist, __binarize, __suffix
  #首先计算所选特征集中每个特征的每个事件的频率
  term_freq = defaultdict(lambda: defaultdict(int))
  term_index = defaultdict(Enumerator())

  for bucket in buckets:
		for path in os.listdir(bucket):
			if path.endswith(__suffix):
				for key, event_id, count in unmarshal_iter(os.path.join(bucket,path)):
					# 只选择列出的特征
					if key in __features:
						term_index[key]
						term_freq[key][event_id] += count

  num_term = len(term_index)
  num_event = len(__dist)

  cm_pos = numpy.zeros((num_term, num_event), dtype='int')

  for term,term_id in term_index.iteritems():
    # 更新事件矩阵
    freq = term_freq[term]
    for event_id, count in freq.iteritems():
      cm_pos[term_id, event_id] = count
  cm_neg = __dist - cm_pos
  cm = numpy.dstack((cm_neg, cm_pos))

  if not __binarize:
    # 非二值化的事件空间
    x = cm.sum(axis=1)
    term_w = x / x.sum(axis=1)[:, None].astype(float)

    #包含该术语与否的熵
    e = entropy(cm, axis=1)

    # 计算到的IG
    ig = entropy(__dist) - (term_w * e).sum(axis=1)

  else:
    ig = list()
    for event_id in xrange(num_event):
      num_doc = __dist.sum()
      prior = numpy.array((num_doc - __dist[event_id], __dist[event_id]), dtype=float) / num_doc

      cm_bin = numpy.zeros((num_term, 2, 2), dtype=int) # (term, p(term), p(lang|term))
      cm_bin[:,0,:] = cm.sum(axis=1) - cm[:,event_id,:]
      cm_bin[:,1,:] = cm[:,event_id,:]

      e = entropy(cm_bin, axis=1)
      x = cm_bin.sum(axis=1)
      term_w = x / x.sum(axis=1)[:, None].astype(float)

      ig.append( entropy(prior) - (term_w * e).sum(axis=1) )
    ig = numpy.vstack(ig)

  terms = sorted(term_index, key=term_index.get)
  return terms, ig

#计算IG
def compute_IG(bucketlist, features, dist, binarize, suffix, job_count=None):
  pass_IG_args = (features, dist, binarize, suffix)

  num_chunk = len(bucketlist)
  weights = []
  terms = []

  with MapPool(job_count, setup_pass_IG, pass_IG_args) as f:
    pass_IG_out = f(pass_IG, bucketlist)

    for i, (t, w) in enumerate(pass_IG_out):
      weights.append(w)
      terms.extend(t)
      print "processed chunk (%d/%d) [%d terms]" % (i+1, num_chunk, len(t))

  if binarize:
    weights = numpy.hstack(weights).transpose()
  else:
    weights = numpy.concatenate(weights)
  terms = ["".join(t) for t in terms]

  return zip(terms, weights)

#从包含item，count的文件中去读分布
def read_dist(path):
  with open(path) as f:
    reader = csv.reader(f)
    return numpy.array(zip(*reader)[1], dtype=int)
