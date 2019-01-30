#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: parse_log.py

import os
import re
import csv
import simplejson as json
import numpy as np
from pprint import pprint
from types import GeneratorType
from collections import defaultdict
from itertools import groupby


Log_File = "test.1901272312.log"


def _iter_split(origin, n):
    """ 将一个列表以n个元素为一个单元进行均分，返回嵌套列表 """
    if isinstance(origin, list):
        return [origin[i:i+n] for i in range(0,len(origin),n)]
    elif isinstance(origin, GeneratorType): # 如果是生成器
        def gen_func(origin): # 将 yield 封装！ 否则无法正常 return
            listFragment = []
            for ele in origin:
                listFragment.append(ele)
                if len(listFragment) >= n:
                    yield listFragment.copy()
                    listFragment.clear()
            if listFragment: # 不到 n 就结束
                yield listFragment
        return gen_func(origin)
    else:
        raise TypeError("illegal type %s for split !" % type(origin))


def _time_to_ms(t):
    _t, unit = t.split()
    _t = float(_t)
    if unit == "s":
        _t *= 1000
    return _t

def _time_to_s(t):
    _t, unit = t.split()
    _t = float(_t)
    if unit == "ms":
        _t /= 1000
    return _t


_times_to_ms   = lambda it: [_time_to_ms(t) for t in it]
_times_to_s    = lambda it: [_time_to_s(t)  for t in it]
_strs_to_float = lambda it: [float(s) for s in it]
_round_float   = lambda f: round(f, 4)
_round_floats  = lambda it: [_round_float(f) for f in it]
_get_mean_std  = lambda it: _round_floats([np.mean(it), np.std(it)])


def parse_test_log():

    regex_log_seg = re.compile(\
    r"=== (?P<feature>feature\d{1})\s*(?:level=(?P<level>\d{1}))* ===\n" + \
    r"\[log\] Function 'get_XYlist' Start\n" + \
    r"\[log\] Function 'get_XYlist' Done\. Cost (?P<cost1>\S+ (?:s|ms))\n" + \
    r"\[log\] Function 'train' Start\n" + \
    r"\[log\] Function 'train' Done\. Cost (?P<cost2>\S+ (?:s|ms))\n" + \
    r"\[log\] Function 'predict' Start\n" + \
    r"\[log\] Function 'predict' Done\. Cost (?P<cost3>\S+ (?:s|ms))\n" + \
    r"(?P<score1>\S+) (?P<score2>\S+)\n", re.S)

    regex_clf_seg = re.compile(r"<class 'sklearn\.(?:.+)\.(?P<class>\S+)'>")


    with open(Log_File, "r", encoding="utf-8") as fp:
        content = fp.read()

    clfs = regex_clf_seg.findall(content)
    logs = regex_log_seg.findall(content)

    _res = defaultdict(list)
    for clf, log in zip(clfs, _iter_split(logs, 8)):
        _res[clf].append(log)

    for k, v in _res.items():
        _logs = [list(zip(*it)) for it in list(zip(*v))]
        _v = []
        for feature, level,  get_XYlist_cost, train_cost, predict_cost, score1, score2 in _logs:

            feature, level = map(lambda it: it[0], [feature, level])
            get_XYlist_cost, train_cost, predict_cost = map(_times_to_ms, [get_XYlist_cost, train_cost, predict_cost])
            score1, score2 = map(_strs_to_float, [score1,score2])

            get_XYlist_cost_avg, get_XYlist_cost_std = _get_mean_std(get_XYlist_cost)
            train_cost_avg, train_cost_std           = _get_mean_std(train_cost)
            predict_cost_avg, predict_cost_std       = _get_mean_std(predict_cost)
            score1_avg, score1_std                   = _get_mean_std(score1)
            score2_avg, score2_std                   = _get_mean_std(score2)

            _v.append([
                feature,
                level,
                get_XYlist_cost_avg,
                get_XYlist_cost_std,
                train_cost_avg,
                train_cost_std,
                predict_cost_avg,
                predict_cost_std,
                score1_avg,
                score1_std,
                score2_avg,
                score2_std,])

        _res[k] = _v

    #res = [[clf, *_log] for clf, _logs in _res.items() for _log in _logs]
    res = []
    for clf, _logs in _res.items():
        for _log in _logs:
            res.append([clf, *_log])

    '''with open(Log_File[:-4] + ".json", "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=4)'''

    headers = ("classifier", "feature", "level", "get_XYlist_cost_avg", "get_XYlist_cost_std",\
                "train_cost_avg", "train_cost_std", "predict_cost_avg", "predict_cost_std",\
                "accuracy_avg", "accuracy_std", "accuracy_ignore_case_avg", "accuracy_ignore_case_std")
    with open(Log_File[:-4] + ".csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, headers)
        writer.writerow(headers)
        writer.writerows(res)

    #pprint(logs)
    #print(len(clfs),len(logs))


if __name__ == '__main__':
    parse_test_log()
'''
    r"=== feature(?P<featureID>\d{1}) (level=(?P<level>)*) ===" +
    r"",re.S)

    <class 'sklearn.neighbors.classification.KNeighborsClassifier'>
=== feature1 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 4.107926 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 1.443560 s
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 16.559668 s
0.9834415584415584 0.9925324675324675
=== feature2 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 7.669813 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 96.267700 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 430.669069 ms
0.9831168831168832 0.9948051948051948
=== feature3 level=1 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 61.693099 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 935.328245 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 4.369326 s
0.9831168831168832 0.9944805194805195
=== feature3 level=2 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 52.722810 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 308.814287 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 1.540697 s
0.9876623376623377 0.9944805194805195
=== feature4 level=1 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 60.267409 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 34.539461 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 377.514362 ms
0.9821428571428571 0.9922077922077922
=== feature4 level=2 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 45.786341 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 33.688784 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 408.865929 ms
0.9840909090909091 0.9925324675324675
=== feature5 level=1 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 89.669401 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 492.083073 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 2.407549 s
0.9866883116883117 0.9961038961038962
=== feature5 level=2 ===
[log] Function 'get_XYlist' Start
[log] Function 'get_XYlist' Done. Cost 75.783831 s
[log] Function 'train' Start
[log] Function 'train' Done. Cost 318.889380 ms
[log] Function 'predict' Start
[log] Function 'predict' Done. Cost 1.968642 s
0.9818181818181818 0.9964285714285714
'''