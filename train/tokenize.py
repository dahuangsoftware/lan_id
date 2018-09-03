#!/usr/bin/env python
#coding=utf-8

MIN_NGRAM_ORDER = 1 # smallest order of n-grams to consider
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOP_DOC_FREQ = 15000 # number of tokens to consider for each order
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os, sys, argparse
import csv
import shutil
import marshal
import multiprocessing as mp
import random
import atexit
import gzip
import tempfile
from itertools import tee 
from collections import defaultdict, Counter
from common import makedir, chunk, MapPool

class NGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    min_order = self.min_order
    max_order = self.max_order
    t = tee(seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = ''.join(tn.next() for tn in t)
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield token[:n+1]
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield token[a:a+b]

class WordNGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    _seq = str.split(seq)
    min_order = self.min_order
    max_order = self.max_order
    t = tee(_seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = [tn.next() for tn in t]
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield ' '.join(token[:n+1])
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield ' '.join(token[a:a+b])

@atexit.register
def cleanup():
  global b_dirs, complete
  try:
    if not complete:
      for d in b_dirs:
        shutil.rmtree(d)
  except NameError:
    # Failed before globals defined, nothing to clean
    pass
  except OSError:
    # sometimes we try to clean up files that are not there
    pass

def setup_pass_tokenize(tokenizer, b_dirs, sample_count, sample_size, term_freq, line_level):
  global __tokenizer, __b_dirs, __sample_count, __sample_size, __term_freq, __line_level
  __tokenizer = tokenizer
  __b_dirs = b_dirs
  __sample_count = sample_count
  __sample_size = sample_size
  __term_freq = term_freq
  __line_level = line_level

#计算频率
def pass_tokenize(chunk_items):
  global __maxorder, __b_dirs, __tokenizer, __sample_count, __sample_size, __term_freq, __line_level
  
  extractor = __tokenizer
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))

  for domain_id, lang_id, path in chunk_items:
    with open(path) as f:
      if __sample_count:
        text = f.read()
        poss = max(1,len(text) - __sample_size)
        count = min(poss, __sample_count)
        offsets = random.sample(xrange(poss), count)
        for offset in offsets:
          tokens = extractor(text[offset: offset+__sample_size])
          if __term_freq:
            tokenset = Counter(tokens)
          else:
            tokenset = Counter(set(tokens))
          for token, count in tokenset.iteritems():
            term_lng_freq[token][lang_id] += count
            term_dom_freq[token][domain_id] += count
      elif __line_level:
        for line in f:
          tokens = extractor(line)
          if __term_freq:
            tokenset = Counter(tokens)
          else:
            tokenset = Counter(set(tokens))
          for token, count in tokenset.iteritems():
            term_lng_freq[token][lang_id] += count
            term_dom_freq[token][domain_id] += count
          
      else:
        #文档标记
        tokens = extractor(f.read())
        if __term_freq:
          # 词项频率
          tokenset = Counter(tokens)
        else:
          # 计算文档频率
          tokenset = Counter(set(tokens))
        for token, count in tokenset.iteritems():
          term_lng_freq[token][lang_id] += count
          term_dom_freq[token][domain_id] += count
  __procname = mp.current_process().name
  b_freq_lang = [gzip.open(os.path.join(p,__procname+'.lang'),'a') for p in __b_dirs]
  b_freq_domain = [gzip.open(os.path.join(p,__procname+'.domain'),'a') for p in __b_dirs]

  for term in term_lng_freq:
    bucket_index = hash(term) % len(b_freq_lang)
    for lang, count in term_lng_freq[term].iteritems():
      b_freq_lang[bucket_index].write(marshal.dumps((term, lang, count)))
    for domain, count in term_dom_freq[term].iteritems():
      b_freq_domain[bucket_index].write(marshal.dumps((term, domain, count)))

  # 关闭文件
  for f in b_freq_lang + b_freq_domain:
    f.close()

  return len(term_lng_freq)

def build_index(items, tokenizer, outdir, buckets=NUM_BUCKETS, 
        jobs=None, chunksize=CHUNKSIZE, sample_count=None, 
        sample_size=None, term_freq=False, line_level=False):
  global b_dirs, complete

  #判断是否删除标记的文件
  complete = False 

  if jobs is None:
    jobs = mp.cpu_count() + 4

  b_dirs = [ os.path.join(outdir,"bucket{0}".format(i)) for i in range(buckets) ]

  for d in b_dirs:
    os.mkdir(d)

  # PASS 1: 将文档分为几组
  chunk_size = max(1,min(len(items) / (jobs * 2), chunksize))
  item_chunks = list(chunk(items, chunk_size))
  pass_tokenize_globals = (tokenizer, b_dirs, sample_count, sample_size, term_freq, line_level)

  with MapPool(jobs, setup_pass_tokenize, pass_tokenize_globals) as f:
    pass_tokenize_out = f(pass_tokenize, item_chunks)
    doc_count = defaultdict(int)
    chunk_count = len(item_chunks)
    print "chunk size: {0} ({1} chunks)".format(chunk_size, chunk_count)
    print "job count: {0}".format(jobs)
    if sample_count:
      print "sampling-based tokenization: size {0} count {1}".format(sample_size, sample_count)
    else:
      print "whole-document tokenization"

    for i, keycount in enumerate(pass_tokenize_out):
      print "tokenized chunk (%d/%d) [%d keys]" % (i+1,chunk_count, keycount)

  complete = True

  return b_dirs
