#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function

try:
    input = raw_input
except NameError:
    pass

# Defaults for inbuilt server
HOST = None  # leave as none for auto-detect
PORT = 9008
FORCE_WSGIREF = False
NORM_PROBS = False  # Normalize output probabilities.

import base64
import bz2
import json
import optparse
import sys
import logging
import numpy as np
import os
from wsgiref.simple_server import make_server
from wsgiref.util import shift_path_info
from collections import defaultdict

try:
    from urllib.parse import parse_qs
except ImportError:
    from urlparse import parse_qs

try:
    from cPickle import loads
except ImportError:
    from pickle import loads

logger = logging.getLogger(__name__)

# 下面定义的便捷方法将在首次调用时初始化。
identifier = None

 #语言识别类，实际上实现语种识别
class LanguageIdentifier(object):

    # 从字符串获取model
    @classmethod
    def from_modelstring(cls, string, *args, **kwargs):
        b = base64.b64decode(string)
        z = bz2.decompress(b)
        model = loads(z)
        nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
        nb_numfeats = int(len(nb_ptc) / len(nb_pc))

        # reconstruct pc and ptc
        nb_pc = np.array(nb_pc)
        nb_ptc = np.array(nb_ptc).reshape(nb_numfeats, len(nb_pc))

        return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

    # 从路径构建模型
    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        with open(path) as f:
            return cls.from_modelstring(f.read().encode(), *args, **kwargs)

    def __init__(self, nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output,
                 norm_probs=NORM_PROBS):
        self.nb_ptc = nb_ptc
        self.nb_pc = nb_pc
        self.nb_numfeats = nb_numfeats
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        self.tk_output = tk_output

        if norm_probs:
            def norm_probs(pd):
                with np.errstate(over='ignore'):
                    pd_exp = np.exp(pd)
                    if pd_exp.sum()!=0:
                        pd = pd_exp / pd_exp.sum()
                return pd
        else:
            def norm_probs(pd):
                return pd

        self.norm_probs = norm_probs
        # 保持对完整模型的引用
        self.__full_model = nb_ptc, nb_pc, nb_classes

    # 限制语种的范围
    def set_languages(self, langs=None):
        nb_ptc, nb_pc, nb_classes = self.__full_model

        if langs is None:
            self.nb_classes = nb_classes
            self.nb_ptc = nb_ptc
            self.nb_pc = nb_pc

        else:
            # 限制语言集合的范围并相应地处理数组用于加速
            for lang in langs:
                if lang not in nb_classes:
                    raise ValueError("Unknown language code %s" % lang)

            subset_mask = np.fromiter((l in langs for l in nb_classes), dtype=bool)
            self.nb_classes = [c for c in nb_classes if c in langs]
            self.nb_ptc = nb_ptc[:, subset_mask]
            self.nb_pc = nb_pc[subset_mask]

    # 将实例映射到训练模型得特征空间
    def instance2fv(self, text):
        if (sys.version_info > (3, 0)):
            # Python3
            if isinstance(text, str):
                text = text.encode('utf8')
        else:
            # Python2
            if isinstance(text, unicode):
                text = text.encode('utf8')
            # 将文本转化成ascii码系列
            # Convert the text to a sequence of ascii values
            text = map(ord, text)

        arr = np.zeros((self.nb_numfeats,), dtype='uint32')

        # Count the number of times we enter each state
        state = 0
        statecount = defaultdict(int)
        for letter in text:
            state = self.tk_nextmove[(state << 8) + letter]
            statecount[state] += 1

        # Update all the productions corresponding to the state
        for state in statecount:
            for index in self.tk_output.get(state, []):
                arr[index] += statecount[state]

        return arr

    # 计算log概率
    def nb_classprobs(self, fv):
        # compute the partial log-probability of the document given each class
        pdc = np.dot(fv, self.nb_ptc)
        # compute the partial log-probability of the document in each class
        pd = pdc + self.nb_pc
        return pd

    # 对一个输入的实例进行识别
    def classify(self, text):
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        cl = np.argmax(probs)
        conf = float(probs[cl])
        pred = str(self.nb_classes[cl])
        return pred, conf

    # 返回一个可能的语言集合
    def rank(self, text):
        """
        Return a list of languages in order of likelihood.
        """
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        return [(str(k), float(v)) for (v, k) in sorted(zip(probs, self.nb_classes), reverse=True)]
#读取国家名称
def readname():
    f = open("name.txt")
    name={}
    line = f.readline().split("\n")[0]
    line1 = f.readline().split("\n")[0]
    while line:
        name[line]=line1
        line = f.readline().split("\n")[0]
        line1 = f.readline().split("\n")[0]
    return name

#函数运行的主逻辑
def main():
    # 声明为全局变量
    global identifier
    def _process(text):
        if False:
            payload = identifier.rank(text)
        else:
            payload = identifier.classify(text)
        return payload
    import sys
    # 看是否是终端输入
    if os.path.exists("train/data.model/model"):
        identifier = LanguageIdentifier.from_modelpath("train/data.model/model",norm_probs=True)
        print("使用训练的model")
    else:
        identifier = LanguageIdentifier.from_modelpath("model",norm_probs=True)
        print("使用默认的model")
    name = readname()
    while True:
        try:
            print(">>>请输入数据:")
            print(">>>", end=' ')
            text = input()
        except Exception as e:
            print(e)
            break
        result = identifier.classify(text)[0]
        list = identifier.rank(text)
        print("The Result are as follows")
        if name.has_key(result):
            result= name[result]
        print(result)
        print("The most likely languages are:")
        for value in list[0:5]:
            if name.has_key(value[0]):
                print(value[0], name[value[0]], value[1])
            else:
                print(value[0],value[1])

if __name__ == "__main__":
    main()