#!/usr/bin/env python
#coding=utf-8
#一些其他类中用到的基本的方法
from itertools import islice
import marshal
import tempfile
import gzip
import shutil
import csv
import os, errno
import numpy
from itertools import imap
from contextlib import contextmanager, closing
import multiprocessing as mp

#枚举对象
class Enumerator(object):
  def __init__(self, start=0):
    self.n = start

  def __call__(self):
    retval = self.n
    self.n += 1
    return retval

#将序列分为不超过设定大小的块
def chunk(seq, chunksize):
  seq_iter = iter(seq)
  while True:
    chunk = tuple(islice(seq_iter, chunksize))
    if not chunk: break
    yield chunk

def unmarshal_iter(path):
  tmpfolder = tempfile.mkdtemp()
  try:
      tmpfile = os.path.join(tmpfolder, "temp")
      with open(tmpfile, "wb") as binfile:
          binfile.write(gzip.open(path, 'rb').read())

      with open(tmpfile, "rb") as binfile:
          while True:
              try:
                  yield marshal.load(binfile)
              except EOFError:
                  break
  finally:
      if tmpfolder and os.path.isdir(tmpfolder):
          shutil.rmtree(tmpfolder)

#生成目录
def makedir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

def write_weights(weights, path, sort_by_weight=False):
  w = dict(weights)
  with open(path, 'w') as f:
    writer = csv.writer(f)
    if sort_by_weight:
      try:
        key_order = sorted(w, key=w.get, reverse=True)
      except ValueError:
        key_order = sorted(w)
    else:
      key_order = sorted(w)

    for k in key_order:
      row = [repr(k)]
      try:
        row.extend(w[k])
      except TypeError:
        row.append(w[k])
      writer.writerow(row)

def read_weights(path):
  with open(path) as f:
    reader = csv.reader(f)
    retval = dict()
    for row in reader:
      key = eval(row[0])
      val = numpy.array( [float(v) if v != 'nan' else 0. for v in row[1:]] )
      retval[key] = val
  return retval

#读取一些特征
def read_features(path):
  with open(path) as f:
    return map(eval, f)

#存储选取的特征
def write_features(features, path):
  with open(path,'w') as f:
    for feat in features:
      print >>f, repr(feat)

#构建索引，返回一个字典
def index(seq):
  return dict((k,v) for (v,k) in enumerate(seq))

#如果只有一个作业，则不适用批处理模式
@contextmanager
def MapPool(processes=None, initializer=None, initargs=None, maxtasksperchild=None, chunksize=1):
  if processes is None:
    processes = mp.cpu_count() + 4

  if processes > 1:
    with closing( mp.Pool(processes, initializer, initargs, maxtasksperchild)) as pool:
      f = lambda fn, chunks: pool.imap_unordered(fn, chunks, chunksize=chunksize)
      yield f
  else:
    if initializer is not None:
      initializer(*initargs)
    f = imap
    yield f

  if processes > 1:
    pool.join()
