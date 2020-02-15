#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: const.py
# Created Date: Wednesday, January 8th 2020, 12:28:16 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

from _internal import mkdir, absp


CACHE_DIR = absp("./cache/")
LOG_DIR = absp("./log/")
MODEL_DIR = absp("./model/")
DOWNLOAD_DIR = absp("./download/")
DATASET_DIR = absp("./dataset/")
RAW_DATASET_DIR = absp("./dataset/raw/")
SEG_DATASET_DIR = absp("./dataset/seg/")
DECODED_DATASET_DIR = absp("./dataset/decoded/")
TRASH_DATASET_DIR = absp("./dataset/trash/")
DATASET_FILE = absp("./dataset/captcha.bz2")
DATASET_CUSTOM_FILE = absp("./dataset/captcha.user.bz2")
CNN_MODEL_FILE = absp("./model/cnn.pt.gz")

mkdir(CACHE_DIR)
mkdir(LOG_DIR)
mkdir(MODEL_DIR)
mkdir(DOWNLOAD_DIR)
mkdir(DATASET_DIR)
mkdir(RAW_DATASET_DIR)
mkdir(SEG_DATASET_DIR)
mkdir(DECODED_DATASET_DIR)
mkdir(TRASH_DATASET_DIR)

SEG_SIDE_LENGTH = 22
LABELS_NUM = 55