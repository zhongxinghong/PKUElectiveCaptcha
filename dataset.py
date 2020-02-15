#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: dataset.py
# Created Date: Saturday, January 11th 2020, 3:35:28 am
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os
from io import BytesIO
import bz2
import joblib
from tqdm import tqdm
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from utils import u, md5
from const import DATASET_CUSTOM_FILE, DATASET_FILE, CACHE_DIR, DECODED_DATASET_DIR,\
    SEG_SIDE_LENGTH, LABELS_NUM

class ElectiveCaptchaBaseDataset(Dataset):

    def __init__(self, *args, **kwargs):
        self.X = None
        self.y = None
        self.labels = []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]
    
    def get_label(self, ix):
        return self.labels[ix]
    

class ElectiveCaptchaDatasetFromPackage(ElectiveCaptchaBaseDataset):

    def __init__(self, use_cache=True):
        super().__init__()

        dataset_file = DATASET_CUSTOM_FILE if os.path.exists(DATASET_CUSTOM_FILE) else DATASET_FILE
        print("Use dataset %s" % dataset_file)

        with bz2.open(dataset_file, "rb") as fp:
            raw = fp.read()
            hs = md5(raw)

        cache_file = os.path.join(CACHE_DIR, "%s.captcha.gz" % hs)
        if os.path.exists(cache_file) and use_cache:
            print("Use dataset cache %s" % cache_file)
            X, y, labels = joblib.load(cache_file)
            self.labels = labels
            self.X = X
            self.y = y
            return

        Xlist = []
        ylist = []

        N = SEG_SIDE_LENGTH
        BS = 8
        PAD = BS - (N * N % BS)
        CS = math.ceil(N * N / BS)

        with BytesIO(raw) as fp:

            t = tqdm(desc="decode dataset", total=fp.getbuffer().nbytes)
            while True:
                X = fp.read(CS)
                y = fp.read(1)
                if X == b'' or y == b'':
                    break
                t.update(CS + 1)

                X = np.array([ (ck >> ofs) & 0b1 for ck in X for ofs in range(BS-1, -1, -1) ][:-PAD])
                X = 1 - X
                y = u(y)
                Xlist.append(X)
                ylist.append(y)

            t.close()

        labels = list(sorted(set(ylist)))
        assert len(labels) == LABELS_NUM

        ixs = np.empty(128, dtype=np.uint8)
        for ix, c in enumerate(labels):
            ixs[ord(c)] = ix

        X = np.array(Xlist, dtype=np.float32).reshape(-1, 1, N, N)
        y = np.array([ ixs[ord(c)] for c in ylist ], dtype=np.long)

        self.labels = labels
        self.X = X
        self.y = y

        joblib.dump((X, y, labels), cache_file, compress=9)


class ElectiveCaptchaDatasetFromDecodedFolder(ElectiveCaptchaBaseDataset):

    def __init__(self):
        super().__init__()

        Xlist = []
        ylist = []
        paths = []

        N = SEG_SIDE_LENGTH

        for label in tqdm(os.listdir(DECODED_DATASET_DIR), "load decoded dataset"):
            subdir = os.path.join(DECODED_DATASET_DIR, label)
            y = label.rstrip("_")
            for filename in os.listdir(subdir):
                path = os.path.join(subdir, filename)
                im = cv2.imread(path)
                X = np.array(( im.sum(axis=2) // 3 ) >> 7, dtype=np.uint8)
                X = 1 - X
                paths.append(path)
                Xlist.append(X)
                ylist.append(y)
    
        labels = list(sorted(set(ylist)))
        assert len(labels) == LABELS_NUM

        ixs = np.empty(128, dtype=np.uint8)
        for ix, c in enumerate(labels):
            ixs[ord(c)] = ix

        X = np.array(Xlist, dtype=np.float32).reshape(-1, 1, N, N)
        y = np.array([ ixs[ord(c)] for c in ylist ], dtype=np.long)

        self.labels = labels
        self.paths = paths
        self.X = X
        self.y = y


    def get_path(self, ix):
        return self.paths[ix]
