#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: preprocess.py

import os
import re
from functools import partial
from tqdm import tqdm
from pprint import pprint
from PIL import Image
from util import mkdir, pkl_dump, pkl_load,\
        Raw_Captcha_Dir, Treated_Captcha_Dir, Segs_Dir, Cache_Dir

regex_captcha = re.compile(r'^\w{4}$')
regex_md5 = re.compile(r'^\w{32}$')

def verify_fmt():
    for name in [file[:-4] for file in os.listdir(Raw_Captcha_Dir)]:
        if not regex_captcha.match(name) and not regex_md5.match(name):
            print(name)


WordSpansList_Cache = "wordSpansList.pkl"
WordSpansList_Cache_File = os.path.join(Cache_Dir, WordSpansList_Cache)

PX_White = 255
PX_Black = 0

StepsLayer1 = ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
StepsLayer2 = ((2,2),(2,1),(2,0),(2,-1),(2,-2),(1,2),(1,-2),(0,2),\
               (0,-2),(-1,2),(-1,-2),(-2,2),(-2,1),(-2,0),(-2,-1),(-2,-2))
Steps4  = ((0,1), (0,-1),(1,0), (-1,0))
Steps8  = StepsLayer1
Steps24 = StepsLayer1 + StepsLayer2


def _trim(img, level=1, scope=[1,1,1,1]):
    """ 去掉 level 层边框
        scope -> [left, upper, right, lower]
    """
    assert img.mode == "1"
    border_height = []
    border_width = []
    for k in range(level):
        if scope[0]: border_width.append(k)
        if scope[1]: border_height.append(k)
        if scope[2]: border_width.append(img.width-1-k)
        if scope[3]: border_height.append(img.height-1-k)
    for j in range(img.width):
        for i in range(img.height):
            if i in border_height or j in border_width:
                img.putpixel((j,i), PX_White)
    return img

trim1 = partial(_trim, level=1)
trim2 = partial(_trim, level=2)
trim_not_left  = partial(_trim, scope=[0,1,1,1])
trim_not_upper = partial(_trim, scope=[1,0,1,1])
trim_not_right = partial(_trim, scope=[1,1,0,1])
trim_not_lower = partial(_trim, scope=[1,1,1,0])


def _denoise(img, steps, threshold, repeat):
    """ 去噪模板函数
    """
    assert img.mode == "1"
    for _ in range(repeat):
        for j in range(img.width):
            for i in range(img.height):
                px = img.getpixel((j,i))
                if px == PX_White: # 自身白
                    continue
                count = 0
                for x, y in steps:
                    j2 = j + y
                    i2 = i + x
                    if 0 <= j2 < img.width and 0 <= i2 < img.height: # 边界内
                        if img.getpixel((j2,i2)) == PX_White: # 周围白
                            count += 1
                    else: # 边界外全部视为黑
                        count += 1
                if count >= threshold:
                   img.putpixel((j,i), PX_White)
    return img

denoise8  = partial(_denoise, steps=Steps8,  threshold=6,  repeat=2)
denoise24 = partial(_denoise, steps=Steps24, threshold=20, repeat=2)


def _search_blocks(img, steps, min_block_size):
    """ 找到图像中的所有块
    """
    assert img.mode == "1"

    marked = [[0 for j in range(img.width)] for i in range(img.height)]

    def _is_marked(i,j):
        if marked[i][j]:
            return True
        else:
            marked[i][j] = 1
            return False

    def _is_white_px(i,j):
        return img.getpixel((j,i)) == PX_White

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
                js = [j for j,i in block]
                blocks.append( (block, min(js), max(js)) )
    return blocks

def search_blocks(img, steps=Steps8, min_block_size=9):
    """ 找到图像中所有的块并验证分割合理性
    """
    assert img.mode == "1"
    blocks = _search_blocks(img, steps, min_block_size)
    ### 确保不会超过4 ###
    assert 1 <= len(blocks) <= 4 # 确保不超过4
    '''if len(blocks) < 4: # 没有完全切开，尝试用小范围的 Steps4
        blocks2 = search_blocks(img, steps=Steps4)
        if len(blocks2) <= 4: # 未超过 4 则使用 Steps4的结果
            assert len(blocks2) > 1 # 确保不是1
            return blocks2'''
    ### Steps4 有可能导致部分字母被割裂 ###
    if len(blocks) == 1: # 对于1 的处理
        pass
    return blocks


def split_spans(spans):
    """ 确保 spans 为 4 份
    """
    assert 1 <= len(spans) <= 4
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
    assert len(spans) == 4
    return spans


def _crop(img, spans):
    """ 分割图片
    """
    assert img.mode == "1"
    assert len(spans) == 4
    size = img.height # img.height == 22
    segs = []
    for left, right in spans:
        quadImg = Image.new("1", (size,size), PX_White)
        segImg = img.crop((left, 0, right+1, size)) # left, upper, right, and lower
        quadImg.paste(segImg, ( (size-segImg.width) // 2, 0 )) # a 2-tuple giving the upper left corner
        segs.append(quadImg)
    return segs

def crop(img):
    assert img.mode == "1"
    blocks = search_blocks(img, steps=Steps8)
    spans = [i[1:] for i in blocks]
    spans.sort(key=lambda span: sum(span))
    print(spans)
    spans = split_spans(spans)
    print(spans)
    segs = _crop(img, spans)
    return segs


def work1():
    for file in tqdm(os.listdir(Raw_Captcha_Dir)):
        name = file[:-4]
        if not regex_captcha.match(name):
            continue
        img = Image.open(os.path.join(Raw_Captcha_Dir, file))
        img = img.convert("1")
        #img = trim1(img)
        img = denoise8(img, repeat=1)
        img = denoise24(img, repeat=1)
        img.save(os.path.join(Treated_Captcha_Dir, file))

def work2():
    wordSpansList = []
    for file in tqdm(os.listdir(Treated_Captcha_Dir)):
        img = Image.open(os.path.join(Treated_Captcha_Dir, file))
        img = img.convert("1")
        blocks = search_blocks(img, steps=Steps8)
        spans = [i[1:] for i in blocks]
        wordSpansList.append([file, spans])
    pkl_dump(WordSpansList_Cache_File, wordSpansList)
    '''
    wordSpansList = [len(words) for words in wordSpansList]
    print(wordSpansList.count(1))
    print(wordSpansList.count(2))
    print(wordSpansList.count(3))
    print(wordSpansList.count(4))
    print(wordSpansList.count(5))
    '''

def work3():
    wordSpansList = pkl_load(WordSpansList_Cache_File)
    wordSpansList.sort()
    #pprint(wordSpansList)
    for file, spans in tqdm(wordSpansList):
        img = Image.open(os.path.join(Treated_Captcha_Dir, file))
        img = img.convert("1")
        name = file[:-4]
        assert len(name) == 4
        spans.sort(key=lambda span: sum(span))
        spans = split_spans(spans)
        counter = {}
        segs = _crop(img, spans)
        for segImg, ch in zip(segs, name):
            folder = os.path.join(Segs_Dir, ch)
            mkdir(folder)
            idx = counter[ch] = counter.get(ch, len(os.listdir(folder))) + 1
            segImgName = "{}_{:05d}_{}.jpg".format(ch, idx, name)
            segImg.save(os.path.join(folder, segImgName))

def test_crop(captcha):
    img = Image.open(os.path.join(Treated_Captcha_Dir, "%s.jpg" % captcha))
    img = img.convert("1")
    img.show()
    segs = crop(img)
    for segImg in segs:
        segImg.show()


if __name__ == '__main__':
    #verify_fmt()
    work1()
    work2()
    work3()
    #test_crop("zfiA")