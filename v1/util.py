#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: util.py

import os
import time
import pickle
import hashlib
from functools import wraps

__all__ = [

    "MD5",
    "mkdir",
    "pkl_dump",
    "pkl_load",
    "log",

    "Download_Dir",
    "Raw_Captcha_Dir",
    "Treated_Captcha_Dir",
    "Segs_Dir",
    "Model_Dir",
    "Cache_Dir",

    ]


def _to_bytes(data):
    if isinstance(data, (str,int,float)):
        return str(data).encode("utf-8")
    elif isinstance(data, bytes):
        return data
    else:
        raise TypeError

MD5 = lambda data: hashlib.md5(_to_bytes(data)).hexdigest()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def pkl_dump(file, data):
    with open(file, "wb") as fp:
        pickle.dump(data, fp)
        print("pickle.dump at " + os.path.abspath(file))

def pkl_load(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    print("pickle.load at " + os.path.abspath(file))
    return data

def log(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print("[log] Function '%s' Start" % fn.__name__)
        t1 = time.time()
        res = fn(*args, **kwargs)
        t2 = time.time()
        cost, unit = t2-t1, "s"
        if cost < 1: # -> ms
            cost, unit = cost*1000, "ms"
        print("[log] Function '%s' Done. Cost %f %s" % (fn.__name__, cost, unit))
        return res
    return wrapper

__base_dir = os.path.dirname(__file__)
__absP = lambda path: os.path.abspath(os.path.join(__base_dir, path))

Base_Dir            = __absP(__base_dir)
Download_Dir        = __absP("./download/")
Raw_Captcha_Dir     = __absP("./raw/")
Treated_Captcha_Dir = __absP("./treated/")
Segs_Dir            = __absP("./segs/")
Model_Dir           = __absP("./model/")
Cache_Dir           = __absP("./cache/")

mkdir(Download_Dir)
mkdir(Raw_Captcha_Dir)
mkdir(Treated_Captcha_Dir)
mkdir(Segs_Dir)
mkdir(Model_Dir)
mkdir(Cache_Dir)
