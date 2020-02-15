#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: classifier.py

import os
import time
from functools import partial
from PIL import Image
import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.externals import joblib
from util import Segs_Dir, Model_Dir, log


__all__ = [

    "FeatureExtractor",

    "KNNClf",
    "SVMClf",
    "RandomForestClf",
    "AdaBoostClf",
    "DecisionTreeClf",

    ]

class FeatureExtractor(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def feature1(img):
        """ 遍历全部像素 """
        ary = np.array(img.convert("1"))
        ary = 1 - ary # 反相
        return ary.flatten()

    @staticmethod
    def feature2(img):
        """ feature2 降维 """
        ary = np.array(img.convert("1"))
        ary = 1 - ary # 反相
        return np.concatenate([ary.sum(axis=0), ary.sum(axis=1)])

    @staticmethod
    def feature3(img, level=2):
        """ 考虑临近像素的遍历 """
        ary = np.array(img.convert("1"))
        ary = 1 - ary # 反相
        l = level
        featureVector = []
        for i in range(l, ary.shape[0]-l):
            for j in range(l, ary.shape[1]-l):
                i1,i2,j1,j2 = i-l, i+l+1, j-l, j+l+1
                featureVector.append(np.sum(ary[i1:i2, j1:j2])) # sum block
        return np.array(featureVector)

    @staticmethod
    def feature4(img, level=2):
        """ feature3 降维 """
        ary = Classifier.feature3(img, level)
        s = int(np.sqrt(ary.size))
        assert s**2 == ary.size # 确保为方
        ary.resize((s,s))
        return np.concatenate([ary.sum(axis=0), ary.sum(axis=1)])

    @staticmethod
    def feature5(img, level=2):
        """ feature3 改版，给接近中心的点增加权重

            weight 矩阵例如：
            array([[1, 1, 1, 1, 1],
                   [1, 2, 2, 2, 1],
                   [1, 2, 3, 2, 1],
                   [1, 2, 2, 2, 1],
                   [1, 1, 1, 1, 1]])
        """
        ary = np.array(img.convert("1"))
        ary = 1 - ary # 反相
        l = level
        s = size = 2 * l + 1
        weight = np.zeros(s**2,dtype=np.int).reshape((s,s))
        for k in range(l+1):
            mask = np.array([k<=i<s-k and k<=j<s-k for i in range(s) for j in range(s)]).reshape((s,s))
            weight[mask] += (k + 1)**2 # 等比数列
        featureVector = []
        for i in range(l, ary.shape[0]-l):
            for j in range(l, ary.shape[1]-l):
                i1,i2,j1,j2 = i-l, i+l+1, j-l, j+l+1
                featureVector.append(np.sum(ary[i1:i2, j1:j2]*weight)) # sum block with weight
        return np.array(featureVector)


class Classifier(FeatureExtractor):

    Model_File = ""
    Model_Compress = 9
    Default_Feature = None

    def __init__(self, *args, **kwargs):
        self.clf = None
        raise NotImplementedError

    @staticmethod
    #@log
    def get_XYlist(feature):
        Xlist = []
        Ylist = []
        for charDir in os.listdir(Segs_Dir):
            for file in os.listdir(os.path.join(Segs_Dir, charDir)):
                img = Image.open(os.path.join(Segs_Dir, charDir, file))
                featureVector = feature(img)
                Xlist.append(featureVector)
                Ylist.append(charDir)
        return Xlist, Ylist

    @staticmethod
    def split_data(Xlist, Ylist, test_size=0.2, **kwargs):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xlist, Ylist, test_size=test_size, **kwargs)
        return Xtrain, Xtest, Ytrain, Ytest

    #@log
    def train(self, Xtrain, Ytrain, *args, **kwargs):
        return self.clf.fit(Xtrain, Ytrain, *args, **kwargs)

    @log
    def predict(self, Xtest):
        return self.clf.predict(Xtest)

    def report_test(self, Xtest, Ytest):
        Ypredict = self.predict(Xtest)
        score1 = accuracy_score(Ytest, Ypredict)
        score2 = accuracy_score(list(map(str.lower, Ytest)), list(map(str.lower, Ypredict)))
        print(score1, score2)

    def test_feature(self, feature):
        Xlist, Ylist = self.get_XYlist(feature)
        Xtrain, Xtest, Ytrain, Ytest = self.split_data(Xlist, Ylist)
        self.train(Xtrain, Ytrain)
        self.report_test(Xtest, Ytest)

    def test_features(self):
        print(self.clf.__class__)
        print("=== feature1 ===")
        self.__init__()
        self.test_feature(self.feature1)
        print("=== feature2 ===")
        self.__init__()
        self.test_feature(self.feature2)
        print("=== feature3 level=1 ===")
        self.__init__()
        self.test_feature(partial(self.feature3, level=1))
        print("=== feature3 level=2 ===")
        self.__init__()
        self.test_feature(partial(self.feature3, level=2))
        print("=== feature4 level=1 ===")
        self.__init__()
        self.test_feature(partial(self.feature4, level=1))
        print("=== feature4 level=2 ===")
        self.__init__()
        self.test_feature(partial(self.feature4, level=2))
        print("=== feature5 level=1 ===")
        self.__init__()
        self.test_feature(partial(self.feature5, level=1))
        print("=== feature5 level=2 ===")
        self.__init__()
        self.test_feature(partial(self.feature5, level=2))

    @log
    def train_model(self, feature=None):
        feature = feature or self.__class__.Default_Feature
        assert self.Model_File != ""
        assert feature is not None
        Xlist, Ylist = self.get_XYlist(feature)
        model = self.train(Xlist, Ylist)
        file = os.path.join(Model_Dir, self.Model_File)
        joblib.dump(model, file, self.Model_Compress)


class KNNClf(Classifier):

    Model_File = "KNN.model.f5.l1.c1.bz2"
    Model_Compress = 1
    Default_Feature = partial(FeatureExtractor.feature5, level=1)


    def __init__(self, n_jobs=4, **kwargs):
        self.clf = KNeighborsClassifier(n_jobs=n_jobs, **kwargs)

    def test_n_neighbors(self, n_neighbors_list, feature):
        Xlist, Ylist = self.get_XYlist(feature)
        Xtrain, Xtest, Ytrain, Ytest = self.split_data(Xlist, Ylist)
        self.train(Xtrain, Ytrain)
        for n in n_neighbors_list:
            self.clf.n_neighbors = n
            t1 = time.time()
            Ypredict = self.clf.predict(Xtest)
            score1 = accuracy_score(Ytest, Ypredict)
            score2 = accuracy_score(list(map(str.lower, Ytest)), list(map(str.lower, Ypredict)))
            t2 = time.time()
            print("{:3d}\t{:.6f}\t{:.6f}\t{:.4f} ms".format(n, score1, score2, (t2-t1)*1000/len(Xtest)))


    # ns = list(range(1,25))+list(range(25,100,5))

    # knn.test_n_neighbors(ns, partial(knn.feature3, level=1))
    # knn.test_n_neighbors(ns, partial(knn.feature3, level=2))
    # knn.test_n_neighbors(ns, partial(knn.feature5, level=1))
    # knn.test_n_neighbors(ns, partial(knn.feature5, level=2))
    # knn.test_n_neighbors(ns, partial(knn.feature5, level=3))


class SVMClf(Classifier):

    Model_File = "SVM.model.f3.l1.c9.xz"
    Model_Compress = 9
    Default_Feature = partial(FeatureExtractor.feature3, level=1)

    def __init__(self, **kwargs):
        self.clf = SVC(**kwargs)

    def test_C(self, C_list, feature):
        Xlist, Ylist = self.get_XYlist(feature)
        Xtrain, Xtest, Ytrain, Ytest = self.split_data(Xlist, Ylist)
        for C in C_list:
            self.__init__(C=C) # C = 1 最佳
            self.clf.fit(Xtrain, Ytrain)
            t1 = time.time()
            Ypredict = self.clf.predict(Xtest)
            score1 = accuracy_score(Ytest, Ypredict)
            score2 = accuracy_score(list(map(str.lower, Ytest)), list(map(str.lower, Ypredict)))
            t2 = time.time()
            print("{:3d}\t{:.6f}\t{:.6f}\t{:.4f} ms".format(C, score1, score2, (t2-t1)*1000/len(Xtest)))


    #svm = SVMClf()

    #Cs = [i/10 for i in range(1,11)]
    #svm.test_C(Cs, svm.feature2)
    #svm.test_C(Cs, partial(svm.feature3, level=1))


class RandomForestClf(Classifier):

    Model_File = "RandomForest.model.f2.c6.bz2"
    Model_Compress = 6
    Default_Feature = FeatureExtractor.feature2

    def __init__(self, n_jobs=4, **kwargs):
        self.clf = RandomForestClassifier(n_jobs=n_jobs, **kwargs)

class AdaBoostClf(Classifier): # 不适用

    def __init__(self, **kwargs):
        self.clf = AdaBoostClassifier(**kwargs)

class DecisionTreeClf(Classifier):

    def __init__(self, **kwargs):
        self.clf = DecisionTreeClassifier(**kwargs)



def task_test_features():
    for i in range(10):
        KNNClf().test_features()
        SVMClf().test_features()
        RandomForestClf().test_features()
        AdaBoostClf().test_features()
        DecisionTreeClf().test_features()


def task_train_model():
    for Clf in [KNNClf, SVMClf, RandomForestClf]:
        clf = Clf()
        clf.train_model()


if __name__ == '__main__':

    #task_train_model()
    clf = RandomForestClf()
    clf.train_model()