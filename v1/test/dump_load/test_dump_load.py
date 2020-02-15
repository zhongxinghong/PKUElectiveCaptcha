#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: test_dump_load.py

import sys
sys.path.append("../../")

import os
import time
from functools import partial
from classifier import *
from sklearn.externals import joblib
from util import log


Log_File = "test_dump_load.RandomForest.log"
Model_Prefix = "RandomForest.model."
Clf = RandomForestClf
#Feature = partial(FeatureExtractor.feature3, level=1)
Feature = FeatureExtractor.feature2


Exts = (".z",".gz",".bz2",".xz",".lzma")


_get_name = lambda ext, compress: Model_Prefix + str(compress) + ext


def train():
    clf = Clf()
    Xlist, Ylist = clf.get_XYlist(Feature)
    model = clf.train(Xlist, Ylist)
    # print(model)
    return model


@log
def test_dump_ext(compress=3, model=None):
    model = model or train()
    for ext in Exts:
        file = _get_name(ext, compress)
        test_dump_model(model, file, compress)

@log
def test_load_ext(compress=3):
    for ext in Exts:
        file = _get_name(ext, compress)
        model = test_load_model(file, compress)
        os.remove(file)


def test_dump_model(model, file, compress):
    print("Dump model to %s, Compress = %d" % (file, compress))
    t1 = time.time()
    joblib.dump(model, file, compress)
    t2 = time.time()
    ms = (t2-t1)*1000
    print("Done. Cost %f ms." % ms, end=" ")
    print("Size of %s = %f Kb" % (file, os.path.getsize(file)/1024))

def test_load_model(file, compress):
    print("Load model from %s, Compress = %d" % (file, compress))
    t1 = time.time()
    model = joblib.load(file)
    t2 = time.time()
    ms = (t2-t1)*1000
    print("Done. Cost %f ms." % ms, end=" ")
    print("Size of %s = %f Kb" % (file, os.path.getsize(file)/1024))
    return model


@log
def test_dump_compress_level(ext=".z", model=None):
    assert ext in Exts
    model = model or train()
    for compress in range(10):
        file = _get_name(ext, compress)
        test_dump_model(model, file, compress)

@log
def test_load_compress_level(ext=".z"):
    assert ext in Exts
    for compress in range(10):
        file = _get_name(ext, compress)
        model = test_load_model(file, compress)
        os.remove(file)


def test_dump_load():
    fp = open(Log_File,"w",encoding="utf-8")
    sys.stdout = fp
    model = train()
    test_dump_ext(model=model)
    test_load_ext()
    for ext in Exts:
        test_dump_compress_level(ext, model=model)
        test_load_compress_level(ext)
    fp.close()



if __name__ == '__main__':
    test_dump_load()

    #test_dump_compress_level(".bz2")
    #test_load_compress_level(".bz2")