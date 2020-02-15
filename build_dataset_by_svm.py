#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: build_dataset_by_svm.py
# Created Date: Wednesday, January 8th 2020, 1:22:37 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os
import re
import bz2
import math
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

# sklearn 0.19.2
from sklearn.externals import joblib
# import joblib

from utils import md5, u, b
from const import DOWNLOAD_DIR, RAW_DATASET_DIR, SEG_DATASET_DIR, DATASET_FILE, SEG_SIDE_LENGTH


SVM_MODEL = "./model/SVM.model.f3.l1.c9.xz"

reDownload = re.compile(r'^(?P<md5>\w{32})\.jpg$')
reRaw = re.compile(r'^(?P<code>\w{4})_(?P<md5>\w{32})\.jpg$')
reSeg = re.compile(r'^(?P<char>\w{1})_(?P<index>\d{1})_(?P<id>\d{6})_(?P<code>\w{4})\.jpg$')


_STEPS_LAYER_1 = ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
_STEPS_LAYER_2 = ((2,2),(2,1),(2,0),(2,-1),(2,-2),(1,2),(1,-2),(0,2),(0,-2),(-1,2),(-1,-2),(-2,2),(-2,1),(-2,0),(-2,-1),(-2,-2))

STEPS4  = ((0,1),(0,-1),(1,0),(-1,0))
STEPS8  = _STEPS_LAYER_1
STEPS24 = _STEPS_LAYER_1 + _STEPS_LAYER_2

_PX_WHITE = 255
_PX_Black = 0

_DEFAULT_MIN_BLOCK_SIZE = 9


def _assert_image_mode_equals_to_1(img):
    assert img.mode == "1", "image mode must be '1', not %s" % img.mode


def _denoise(img, steps, threshold, repeat):
    """ 去噪函数模板 """
    _assert_image_mode_equals_to_1(img)

    for _ in range(repeat):
        for j in range(img.width):
            for i in range(img.height):
                px = img.getpixel((j,i))
                if px == _PX_WHITE: # 自身白
                    continue
                count = 0
                for x, y in steps:
                    j2 = j + y
                    i2 = i + x
                    if 0 <= j2 < img.width and 0 <= i2 < img.height: # 边界内
                        if img.getpixel((j2,i2)) == _PX_WHITE: # 周围白
                            count += 1
                    else: # 边界外全部视为黑
                        count += 1
                if count >= threshold:
                   img.putpixel((j,i), _PX_WHITE)

    return img


def denoise8(img, steps=STEPS8, threshold=6, repeat=2):
    """ 考虑外一周的降噪 """
    return _denoise(img, steps, threshold, repeat)


def denoise24(img, steps=STEPS24, threshold=20, repeat=2):
    """ 考虑外两周的降噪 """
    return _denoise(img, steps, threshold, repeat)


def _search_blocks(img, steps=STEPS8, min_block_size=_DEFAULT_MIN_BLOCK_SIZE):
    """ 找到图像中的所有块 """
    _assert_image_mode_equals_to_1(img)

    marked = [ [ 0 for j in range(img.width) ] for i in range(img.height) ]


    def _is_marked(i,j):
        if marked[i][j]:
            return True
        else:
            marked[i][j] = 1
            return False


    def _is_white_px(i,j):
        return img.getpixel((j,i)) == _PX_WHITE


    def _queue_search(i,j):
        """ 利用堆栈寻找字母 """
        queue = [(j,i),]
        head = 0
        while head < len(queue):
            now = queue[head]
            head += 1
            for x, y in steps:
                j2 = now[0] + y
                i2 = now[1] + x
                if 0 <= j2 < img.width and 0 <= i2 < img.height:
                    if _is_marked(i2,j2) or _is_white_px(i2,j2):
                        continue
                    queue.append((j2,i2))
        return queue


    blocks = []
    for j in range(img.width):
        for i in range(img.height):
            if _is_marked(i,j) or _is_white_px(i,j):
                continue
            block = _queue_search(i,j)
            if len(block) >= min_block_size:
                js = [ j for j, _ in block ]
                blocks.append( (block, min(js), max(js)) )

    assert 1 <= len(blocks) <= 4, "unexpected number of captcha blocks: %s" % len(blocks)

    return blocks


def _split_spans(spans):
    """ 确保 spans 为 4 份 """
    assert 1 <= len(spans) <= 4, "unexpected number of captcha blocks: %s" % len(spans)

    if len(spans) == 1: # 四等分
        totalSpan = spans[0]
        delta = (totalSpan[1] - totalSpan[0]) // 4
        spans = [
            (totalSpan[0],         totalSpan[0]+delta  ),
            (totalSpan[0]+delta,   totalSpan[0]+delta*2),
            (totalSpan[1]-delta*2, totalSpan[1]-delta  ),
            (totalSpan[1]-delta,   totalSpan[1]        ),
        ]

    if len(spans) == 2: # 三等分较大块
        maxSpan = max(spans, key=lambda span: span[1]-span[0])
        idx = spans.index(maxSpan)
        delta = (maxSpan[1] - maxSpan[0]) // 3
        spans.remove(maxSpan)
        spans.insert(idx,   (maxSpan[0],       maxSpan[0]+delta))
        spans.insert(idx+1, (maxSpan[0]+delta, maxSpan[1]-delta))
        spans.insert(idx+2, (maxSpan[1]-delta, maxSpan[1]      ))

    if len(spans) == 3: # 平均均分较大块
        maxSpan = max(spans, key=lambda span: span[1]-span[0])
        idx = spans.index(maxSpan)
        mid = sum(maxSpan) // 2
        spans.remove(maxSpan)
        spans.insert(idx,   (maxSpan[0], mid))
        spans.insert(idx+1, (mid, maxSpan[1]))

    if len(spans) == 4:
        pass

    return spans


def _crop(img, spans):
    """ 分割图片 """
    _assert_image_mode_equals_to_1(img)
    assert len(spans) == 4, "unexpected number of captcha blocks: %s" % len(spans)

    size = img.height # img.height == 22
    segs = []

    for left, right in spans:
        quadImg = Image.new("1", (size,size), _PX_WHITE)
        segImg = img.crop((left, 0, right+1, size))  # left, upper, right, and lower
        quadImg.paste(segImg, ( (size-segImg.width) // 2, 0 ))  # a 2-tuple giving the upper left corner
        segs.append(quadImg)

    return segs


def crop(img):
    _assert_image_mode_equals_to_1(img)

    blocks = _search_blocks(img, steps=STEPS8)
    spans = [i[1:] for i in blocks]
    spans.sort(key=lambda span: sum(span))
    spans = _split_spans(spans)
    segs = _crop(img, spans)

    return segs, spans


def feature3(im, level=1):
    ary = np.array(im.convert("1"))
    ary = 1 - ary # 反相
    l = level
    vector = []
    for i in range(level, ary.shape[0]-level):
        for j in range(level, ary.shape[1]-level):
            i1, i2, j1, j2 = i-level, i+level+1, j-level, j+level+1
            vector.append(np.sum(ary[i1:i2, j1:j2])) # sum block
    return np.array(vector)


def recognize(svm, im):

    im = im.convert("1")
    
    im = denoise8(im, repeat=1)
    im = denoise24(im, repeat=1)

    segs, spans = crop(im)

    Xlist = [ feature3(sim) for sim in segs ]
    ylist = svm.predict(Xlist)

    code = "".join(ylist)

    return code


def build_raw_dataset():

    ignored = { reRaw.match(x).group(2) for x in os.listdir(RAW_DATASET_DIR) }
    svm = joblib.load(SVM_MODEL)

    for filename in tqdm(os.listdir(DOWNLOAD_DIR), "build raw dataset"):
        
        mat = reDownload.match(filename)
        if mat is None:
            continue

        hs = mat.group(1)
        if hs in ignored:
            continue

        p1 = os.path.join(DOWNLOAD_DIR, filename)

        try:
            im = Image.open(p1)
        except IOError as e:
            if "cannot identify image file" in e.args[0]:
                os.remove(p1)
                continue
            raise e
        
        try:
            code = recognize(svm, im)
        except AssertionError as e:
            if "unexpected number of captcha blocks" in e.args[0]:
                os.remove(p1)
                continue
            raise e

        filename = "%s_%s.jpg" % (code, hs)
        p2 = os.path.join(RAW_DATASET_DIR, filename)

        shutil.copyfile(p1, p2)
        ignored.add(hs)


def build_seg_dataset():

    OFFSET = ord('0')

    ignored = set()
    cnts = [ 0 for _ in range(ord('z') - OFFSET + 1) ]

    for filename in os.listdir(SEG_DATASET_DIR):
        mat = reSeg.match(filename)
        assert mat is not None
        
        char, id_, code = mat.group(1,3,4)
        id_ = int(id_)

        ignored.add(code)

        key = ord(char) - OFFSET
        if cnts[key] < id_:
            cnts[key] = id_

    for filename in tqdm(os.listdir(RAW_DATASET_DIR), "build seg dataset"):
        
        mat = reRaw.match(filename)
        assert mat is not None
        
        code = mat.group(1)
        if code in ignored:
            continue

        p0 = os.path.join(RAW_DATASET_DIR, filename)
        im = Image.open(p0)
        
        im = im.convert("1")
    
        im = denoise8(im, repeat=1)
        im = denoise24(im, repeat=1)

        segs, spans = crop(im)

        for ix, (sim, char) in enumerate(zip(segs, code)):
            key = ord(char) - OFFSET
            cnts[key] += 1

            filename = "%s_%d_%06d_%s.jpg" % (char, ix, cnts[key], code)
            path = os.path.join(SEG_DATASET_DIR, filename)

            sim.save(path)
            
        ignored.add(code)
        

def compress_dataset():

    N = SEG_SIDE_LENGTH
    BS = 8

    weight = np.array([ 1 << ofs for ofs in range(BS-1, -1, -1) ], dtype=np.uint8)
    pad = np.zeros((BS - (N * N % BS)), dtype=np.uint8)

    with bz2.open(DATASET_FILE, "wb") as fp:
        for filename in tqdm(sorted(os.listdir(SEG_DATASET_DIR)), "encode dataset"):
            path = os.path.join(SEG_DATASET_DIR, filename)
            im = cv2.imread(path)
            
            X = np.array(( im.sum(axis=2) // 3 ) >> 7, dtype=np.uint8)
            X = np.hstack([ X.flatten(), pad ]).reshape(-1, BS)
            X = (X * weight).sum(axis=1).astype(np.uint8)
            X = bytes(X)
            y = b(filename[0])
            fp.write(X + y)
            
    
def decode_dataset():
    Xlist = []
    ylist = []
    
    N = SEG_SIDE_LENGTH
    BS = 8
    PAD = BS - (N * N % BS)
    CS = math.ceil(N * N / BS)

    print("decode dataset ...")
    
    with bz2.open(DATASET_FILE, "rb") as fp:
        while True:
            X = fp.read(CS)
            y = fp.read(1)
            if X == b'' or y == b'':
                break

            X = [ (ck >> ofs) & 0b1 for ck in X for ofs in range(BS-1, -1, -1) ][:-PAD]
            X = np.array(X).reshape(N, N).astype(np.uint8)
            y = u(y)
            Xlist.append(X)
            ylist.append(y)

    X = np.array(Xlist)
    y = np.array(ylist)
    
    return X, y
    

def main():
    build_raw_dataset()
    build_seg_dataset()
    compress_dataset()
    # X, y = decode_dataset()
    # print(X[0], y[0], X[0].dtype, y[0].dtype)
    # print(X.shape, y.shape, X.dtype, y.dtype)


if __name__ == "__main__":
    main()
