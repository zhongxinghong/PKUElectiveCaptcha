#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: parse_log.py

import os
import re
import simplejson as json
from pprint import pprint
from test_dump_load import Model_Prefix, Log_File


def parse_test_dump_load():

    regex_func_seg = re.compile(\
    r"\[log\] Function '(?P<fn>\S+)' Start\n" + \
    r"(?P<logs>.*?)\n" + \
    r"\[log\] Function '(?P=fn)' Done\. Cost (?P<cost>\S+ (?:ms|s))\n", re.S)

    regex_log_seg = re.compile(\
    r"(?:Dump|Load) model (?:to|from) (?P<file>" + Model_Prefix + r"(?P<compress>\d{1})" + \
    r"(?P<ext>\.z|\.gz|\.bz2|\.xz|\.lzma)), Compress = (?P=compress)\n" + \
    r"Done\. Cost (?P<cost>\S+ ms)\. Size of (?P=file) = (?P<size>\S+) Kb\n")

    func_seg_fields = ("fn","logs","cost")
    log_seg_fields = ("file","compress","ext","cost","size")

    with open(Log_File, "r", encoding="utf-8") as fp:
        content = fp.read()

    outJson = []
    for fn, logs, cost in regex_func_seg.findall(content):
        _logs = []
        for res in regex_log_seg.findall(logs):
            _logs.append(dict(zip(log_seg_fields,res)))
        outJson.append(dict(zip(func_seg_fields, [fn,_logs,cost])))

    with open(Log_File[:-4] + ".json", "w", encoding="utf-8") as fp:
        json.dump(outJson, fp, indent=4)

if __name__ == '__main__':
    parse_test_dump_load()