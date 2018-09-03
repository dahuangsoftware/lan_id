#!/usr/bin/env python
#coding=utf-8
MAX_CHUNK_SIZE = 100
NUM_BUCKETS = 64

import base64, bz2, cPickle
import os, sys, argparse, csv
import array
import numpy as np
import tempfile
import marshal
import atexit, shutil
import multiprocessing as mp
import gzip
from collections import deque, defaultdict
from contextlib import closing
from common import chunk, unmarshal_iter, read_features, index, MapPool

#返回每一个状态的计数
def state_trace(text):
  global __nm_arr
  c = defaultdict(int)
  state = 0
  for letter in map(ord,text):
    state = __nm_arr[(state << 8) + letter]
    c[state] += 1
  return c

def setup_pass_tokenize(nm_arr, output_states, tk_output, b_dirs, line_level):
  global __nm_arr, __output_states, __tk_output, __b_dirs, __line_level
  __nm_arr = nm_arr
  __output_states = output_states
  __tk_output = tk_output
  __b_dirs = b_dirs
  __line_level = line_level

#拆分文档，对于每一个特征计数
def pass_tokenize(arg):
  global __output_states, __tk_output, __b_dirs, __line_level
  chunk_id, chunk_paths = arg
  term_freq = defaultdict(int)

  # 对每一个文档进行计数
  doc_count = 0
  labels = []
  for label, path in chunk_paths:
    with open(path) as f:
      if __line_level:
        for text in f:
          count = state_trace(text)
          for state in (set(count) & __output_states):
            for f_id in __tk_output[state]:
              term_freq[doc_count, f_id] += count[state]
          doc_count += 1
          labels.append(label)

      else:
        text = f.read()
        count = state_trace(text)
        for state in (set(count) & __output_states):
          for f_id in __tk_output[state]:
            term_freq[doc_count, f_id] += count[state]
        doc_count += 1
        labels.append(label)

  # 将计数分发到各个桶中
  __procname = mp.current_process().name
  __buckets = [gzip.open(os.path.join(p,__procname+'.index'), 'a') for p in __b_dirs]
  bucket_count = len(__buckets)
  for doc_id, f_id in term_freq:
    bucket_index = hash(f_id) % bucket_count
    count = term_freq[doc_id, f_id]
    item = ( f_id, chunk_id, doc_id, count )
    __buckets[bucket_index].write(marshal.dumps(item))

  for f in __buckets:
    f.close()

  return chunk_id, doc_count, len(term_freq), labels

def setup_pass_ptc(cm, num_instances, chunk_offsets):
  global __cm, __num_instances, __chunk_offsets
  __cm = cm
  __num_instances = num_instances
  __chunk_offsets = chunk_offsets

#对于每一个桶返回一个计数对
def pass_ptc(b_dir):
  global __cm, __num_instances, __chunk_offsets
  terms = defaultdict(lambda : np.zeros((__num_instances,), dtype='int'))

  read_count = 0
  for path in os.listdir(b_dir):
    if path.endswith('.index'):
      for f_id, chunk_id, doc_id, count in unmarshal_iter(os.path.join(b_dir, path)):
        index = doc_id + __chunk_offsets[chunk_id]
        terms[f_id][index] = count
        read_count += 1

  f_ids, f_vs = zip(*terms.items())
  fm = np.vstack(f_vs)
  prod = np.dot(fm, __cm)

  return read_count, f_ids, prod

#NB的参数训练
def learn_nb_params(items, num_langs, tk_nextmove, tk_output, temp_path, args):
  global outdir
  print "learning NB parameters on {} items".format(len(items))
  nm_arr = mp.Array('i', tk_nextmove, lock=False)

  if args.jobs:
    tasks = args.jobs * 2
  else:
    tasks = mp.cpu_count() * 2
  chunksize = max(1, min(len(items) / tasks, args.chunksize))

  outdir = tempfile.mkdtemp(prefix="NBtrain-",suffix='-buckets', dir=temp_path)
  b_dirs = [ os.path.join(outdir,"bucket{0}".format(i)) for i in range(args.buckets) ]

  for d in b_dirs:
    os.mkdir(d)

  output_states = set(tk_output)
  
  # 将要处理的所有项目划分为块，并枚举每个块
  item_chunks = list(chunk(items, chunksize))
  num_chunks = len(item_chunks)
  print "about to tokenize {} chunks".format(num_chunks)
  
  pass_tokenize_arg = enumerate(item_chunks)
  pass_tokenize_params = (nm_arr, output_states, tk_output, b_dirs, args.line) 
  with MapPool(args.jobs, setup_pass_tokenize, pass_tokenize_params) as f:
    pass_tokenize_out = f(pass_tokenize, pass_tokenize_arg)
  
    write_count = 0
    chunk_sizes = {}
    chunk_labels = []
    for i, (chunk_id, doc_count, writes, labels) in enumerate(pass_tokenize_out):
      write_count += writes
      chunk_sizes[chunk_id] = doc_count
      chunk_labels.append((chunk_id, labels))
      print "processed chunk ID:{0} ({1}/{2}) [{3} keys]".format(chunk_id, i+1, num_chunks, writes)

  print "wrote a total of %d keys" % write_count

  num_instances = sum(chunk_sizes.values())
  print "processed a total of %d instances" % num_instances

  chunk_offsets = {}
  for i in range(len(chunk_sizes)):
    chunk_offsets[i] = sum(chunk_sizes[x] for x in range(i))

  cm = np.zeros((num_instances, num_langs), dtype='bool')
  for chunk_id, chunk_label in chunk_labels:
    for doc_id, lang_id in enumerate(chunk_label):
      index = doc_id + chunk_offsets[chunk_id]
      cm[index, lang_id] = True

  pass_ptc_params = (cm, num_instances, chunk_offsets)
  with MapPool(args.jobs, setup_pass_ptc, pass_ptc_params) as f:
    pass_ptc_out = f(pass_ptc, b_dirs)

    def pass_ptc_progress():
      for i,v in enumerate(pass_ptc_out):
        yield v
        print "processed chunk ({0}/{1})".format(i+1, len(b_dirs))

    reads, ids, prods = zip(*pass_ptc_progress())
    read_count = sum(reads)
    print "read a total of %d keys (%d short)" % (read_count, write_count - read_count)

  num_features = max( i for v in tk_output.values() for i in v) + 1
  prod = np.zeros((num_features, cm.shape[1]), dtype=int)
  prod[np.concatenate(ids)] = np.vstack(prods)

  # 数据加一平滑
  ptc = np.log(1 + prod) - np.log(num_features + prod.sum(0))
  nb_ptc = array.array('d')
  for term_dist in ptc.tolist():
    nb_ptc.extend(term_dist)

  pc = np.log(cm.sum(0))
  nb_pc = array.array('d', pc)

  return nb_pc, nb_ptc

@atexit.register
def cleanup():
  global outdir 
  try:
    shutil.rmtree(outdir)
  except NameError:
    pass
  except OSError:
    pass
