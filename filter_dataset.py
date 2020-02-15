#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: filter_dataset.py
# Created Date: Thursday, January 9th 2020, 1:24:55 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os
import re
from io import BytesIO
import bz2
import joblib
from tqdm import tqdm
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _internal import mkdir
from utils import u, b, md5
from const import LOG_DIR, DATASET_FILE, DECODED_DATASET_DIR, DATASET_CUSTOM_FILE,\
    TRASH_DATASET_DIR, CNN_MODEL_FILE, SEG_SIDE_LENGTH
from cnn import ElectiveCaptchaCNN
from cnn import main as train_cnn_model
from dataset import ElectiveCaptchaDatasetFromDecodedFolder


reDecodedSeg = re.compile(r'^(?P<char>\w{1})[_]{1,2}(?P<id>\d{6})\.png$')
reTrashSeg = re.compile(r'^(?P<char>\w{1})_(?P<md5>\w{32})\.png$')
reDecodedSegWithDir = re.compile(r'[/\\](\w[_]{0,1}[/\\]\w[_]{1,2}\S+\.png)$')
reDecodedSegOriginalPath = re.compile(r"^(\w{1})[_]{0,1}[/\\]\1[_]{1,2}\S+\.png$")

DIFF_LOG_FILE = os.path.join(LOG_DIR, "diff.log")


def get_trash_segs_md5():
    
    ignored = set()

    def _ignore_seg(folder, filename):
        src = os.path.join(folder, filename)
        mat = reTrashSeg.match(filename)
        if mat is None:
            mat2 = reDecodedSeg.match(filename)
            assert mat2 is not None
            char = mat2.group(1)
            im = cv2.imread(src)
            hs = md5(im.tobytes())
            filename = "%s_%s.png" % (char, hs)
            dst = os.path.join(folder, filename)
            os.rename(src, dst)
        else:
            hs = mat.group(2)
        ignored.add(hs) 

    for filename in os.listdir(TRASH_DATASET_DIR):
        path = os.path.join(TRASH_DATASET_DIR, filename)
        if os.path.isdir(path):
            for _filename in os.listdir(path):
                _ignore_seg(path, _filename)
        else:
            _ignore_seg(TRASH_DATASET_DIR, filename)
    
    return ignored


def unpack_dataset():

    N = SEG_SIDE_LENGTH
    BS = 8
    PAD = BS - (N * N % BS)
    CS = math.ceil(N * N / BS)

    rgb_white = np.array([255,255,255], dtype=np.uint8)
    rgb_black = np.array([0,0,0], dtype=np.uint8)

    cnts = np.zeros(128, dtype=np.int)

    ignored = get_trash_segs_md5()

    with bz2.open(DATASET_FILE, "rb") as fp:
        raw = fp.read()

    with BytesIO(raw) as fp:
        t = tqdm(desc="unpack dataset", total=fp.getbuffer().nbytes)
        while True:
            X = fp.read(CS)
            y = fp.read(1)
            if X == b'' or y == b'':
                break
            t.update(CS + 1)

            X = np.array([ (ck >> ofs) & 0b1 for ck in X for ofs in range(BS-1, -1, -1) ][:-PAD])
            im = np.array([ rgb_white if b else rgb_black for b in X ], dtype=np.uint8).reshape(N, N, 3)

            y = u(y)
            k = ord(y)
            cnts[k] += 1

            hs = md5(im.tobytes())
            if hs in ignored:                
                continue

            y = y + "_" if y.islower() else y
            folder = os.path.join(DECODED_DATASET_DIR, y)
            filename = "%s_%06d.png" % (y, cnts[k])
            path = os.path.join(folder, filename)

            mkdir(folder)
            cv2.imwrite(path, im)

        t.close()


def pack_dataset():

    N = SEG_SIDE_LENGTH
    BS = 8

    weight = np.array([ 1 << ofs for ofs in range(BS-1, -1, -1) ], dtype=np.uint8)
    pad = np.zeros((BS - (N * N % BS)), dtype=np.uint8)
    ignored = get_trash_segs_md5() 

    with bz2.open(DATASET_CUSTOM_FILE, "wb") as fp:
        for label in tqdm(sorted(os.listdir(DECODED_DATASET_DIR)), "pack dataset"):
            subdir = os.path.join(DECODED_DATASET_DIR, label)
            y = b(label[0])
            for filename in sorted(os.listdir(subdir)):
                path = os.path.join(subdir, filename)
                im = cv2.imread(path).astype(np.uint8)

                hs = md5(im.tobytes())
                if hs in ignored:
                    continue
                
                X = np.array(( im.sum(axis=2) // 3 ) >> 7, dtype=np.uint8)
                X = np.hstack([ X.flatten(), pad ]).reshape(-1, BS)
                X = (X * weight).sum(axis=1).astype(np.uint8)
                X = bytes(X)
                
                fp.write(X + y)


def test_cnn_model_on_decoded_folder():

    BATCH_SIZE = 1024

    model = ElectiveCaptchaCNN()
    model.load_state_dict(joblib.load(CNN_MODEL_FILE))
    
    dataset = ElectiveCaptchaDatasetFromDecodedFolder()

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    test_loss = 0
    correct = 0

    tpp = []

    with torch.no_grad():
        for bix, (data, target) in enumerate(tqdm(test_loader, "test decoded dataset")):
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            same = pred.eq(target.view_as(pred))
            correct += same.sum().item()
            for tix, (t, p, s) in enumerate(zip(target, pred, same)):
                if s.item():
                    continue
                ix = bix * BATCH_SIZE + tix
                t = dataset.get_label(t.item())
                p = dataset.get_label(p.item())
                path = dataset.get_path(ix)
                tpp.append((t,p,path))
                
    test_loss /= len(test_loader.sampler)

    print('\nDecoded dataset: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, 
        correct, 
        len(test_loader.sampler),
        100.0 * correct / len(test_loader.sampler)
    ))

    with open(DIFF_LOG_FILE, "w") as fp:
        for t, p, path in tpp:
            mat = reDecodedSegWithDir.search(path)
            assert mat is not None
            path = mat.group(1)
            if reDecodedSegOriginalPath.match(path) is not None:
                fp.write("[ {} -> {} ] {}\n".format(t, p, path))


def main():

    # unpack_dataset()

    pack_dataset()
    train_cnn_model()
    test_cnn_model_on_decoded_folder()


if __name__ == "__main__":
    main()
