#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: _internal.py
# Created Date: Wednesday, January 8th 2020, 12:29:14 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def absp(*path):
    return os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), *path)))
