#!/usr/bin/env python
#coding=utf-8
#将所有的代码
TRAIN_PROP = 1.0 #用到的训练数据的比例
MIN_DOMAIN = 1 # 一个语言最少包括的domain数
MAX_NGRAM_ORDER = 4 # 要考虑的最大n-gram顺序
TOP_DOC_FREQ = 15000 # 每个n-gram序列考虑的令牌的数量
NUM_BUCKETS = 64 # 所用到的桶个个数，将特征存储在64个桶中
CHUNKSIZE = 50 # 块的大小，表示稳健操作中一次令牌化的数目
FEATURES_PER_LANG = 300 # 对于每一种语言所选取的特征数

import argparse
import os, csv
import numpy
import base64, bz2, cPickle
import shutil
from common import makedir, write_weights, write_features, read_weights, read_features
from index import CorpusIndexer
from tokenize import build_index, NGramTokenizer
from DFfeatureselect import tally, ngram_select
from IGweight import compute_IG
from LDfeatureselect import select_LD_features
from scanner import build_scanner, Scanner
from NBtrain import learn_nb_params

class canshu(object):
  buckets = 64
  chunksize = 50
  corpus = 'lan_id\\data'
  debug = False
  df_feats = None
  df_tokens = 15000
  domain = None
  feats_per_lang = 300
  jobs = None
  lang = None
  ld_feats = None
  line = False
  max_order = 4
  min_domain = 1
  model = None
  no_domain_ig = False
  proportion = 1.0
  sample_count = None
  sample_size = 140
  temp = None
  word = False

if __name__ == "__main__":
  data_path="../data"
  corpus_name = os.path.basename(data_path)
  model_dir = os.path.join('.', corpus_name + '.model')
  makedir(model_dir)
  #语料库index初始化
  #输入：数据路径，最小分类，训练比例，语言，分类数，line
  print "开始进行索引语料库-index......"
  indexer = CorpusIndexer(data_path, min_domain=1, proportion=1.0,
                          langs=None, domains=None, line_level=False)

  # 计算文件，语言和域之间的映射
  lang_dist = indexer.dist_lang
  lang_index = indexer.lang_index
  lang_info = ' '.join(("{0}({1})".format(k, lang_dist[v]) for k, v in lang_index.items()))
  print "langs({0}): {1}".format(len(lang_dist), lang_info)

  domain_dist = indexer.dist_domain
  domain_index = indexer.domain_index
  domain_info = ' '.join(("{0}({1})".format(k, domain_dist[v]) for k, v in domain_index.items()))
  print "domains({0}): {1}".format(len(domain_dist), domain_info)
  print "identified {0} documents".format(len(indexer.items))
  items = sorted(set((d, l, p) for (d, l, n, p) in indexer.items))

  # print("running here 0")
  #给bucket构造路径
  buckets_dir = os.path.join(model_dir, 'buckets')
  makedir(buckets_dir)
  print "建成完毕."
  #计算得到特征
  # Tokenize
  DFfeats = None
  print "will tokenize %d documents" % len(items)
  print "using byte NGram tokenizer, max_order: {0}".format(4)
  tk = NGramTokenizer(1, 4)

  # 首次通过标记化，用于确定特征的DF
  tk_dir = os.path.join(buckets_dir, 'tokenize-pass1')
  makedir(tk_dir)
  b_dirs = build_index(items, tk, tk_dir, 64, None, 50, None,
                         140, False)
  print("running here 0")
  doc_count = tally(b_dirs, None)
  DFfeats = ngram_select(doc_count, 4, 15000)
  shutil.rmtree(tk_dir)

  # 再次仅为所选的特征计数
  DF_scanner = Scanner(DFfeats)
  df_dir = os.path.join(buckets_dir, 'tokenize-pass2')
  makedir(df_dir)
  b_dirs = build_index(items, DF_scanner, df_dir, 64, None, 50)
  b_dirs = [[d] for d in b_dirs]

  # 计算向量值
  domain_dist_vec = numpy.array([domain_dist[domain_index[d]]
                                 for d in sorted(domain_index, key=domain_index.get)], dtype=int)
  lang_dist_vec = numpy.array([lang_dist[lang_index[l]]
                               for l in sorted(lang_index.keys(), key=lang_index.get)], dtype=int)

  # 计算IG权重
  ig_params = [
      ('lang', lang_dist_vec, '.lang', True),
    ]
  print("come here 111")
  ig_params.append(('domain', domain_dist_vec, '.domain', False))

  ig_vals = {}
  for label, dist, suffix, binarize in ig_params:
    print "Computing information gain for {0}".format(label)
    ig = compute_IG(b_dirs, DFfeats, dist, binarize, suffix, None)
    ig_vals[label] = dict((row[0], numpy.array(row[1].flat)) for row in ig)

  # 根据LD选择特征
  features_per_lang = select_LD_features(ig_vals['lang'], ig_vals.get('domain'), 300, ignore_domain=canshu.no_domain_ig)
  LDfeats = reduce(set.union, map(set, features_per_lang.values()))
  print 'selected %d features' % len(LDfeats)

  # 编译LD特征的扫描值
  tk_nextmove, tk_output = build_scanner(LDfeats)

  # 组合成NB模型
  langs = sorted(lang_index, key=lang_index.get)

  nb_classes = langs
  nb_dir = os.path.join(buckets_dir, 'NBtrain')
  makedir(nb_dir)
  nb_pc, nb_ptc = learn_nb_params([(int(l), p) for _, l, p in items], len(langs), tk_nextmove, tk_output, nb_dir,
                                  canshu)

  # 输出模型
  output_path = os.path.join(model_dir, 'model')
  model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  print "wrote model to %s (%d bytes)" % (output_path, len(string))

  # 如果调试关闭，则删除存储桶。 如果提供ldfeats，我们不会生成存储桶。
  if not False and not None:
    shutil.rmtree(df_dir)
    if not None:
      shutil.rmtree(buckets_dir)
