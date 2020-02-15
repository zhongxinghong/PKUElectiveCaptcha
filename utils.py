#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: Wednesday, January 8th 2020, 12:33:21 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import hashlib


def b(s):
    if isinstance(s, (str,int,float)):
        return str(s).encode("utf-8")
    elif isinstance(s, bytes):
        return s
    else:
        raise TypeError("can't convert %s to bytes")

def u(s):
    if isinstance(s, bytes):
        return s.decode("utf-8")
    elif isinstance(s, (str, int, float)):
        return str(s)
    else:
        raise TypeError("can't convert %s to utf-8")


def md5(s):
    return hashlib.md5(b(s)).hexdigest()
