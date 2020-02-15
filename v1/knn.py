#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: knn.py

import os
import time
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from util import Segs_Dir

Xlist = []
Ylist = []
for charDir in os.listdir(Segs_Dir):
    for file in os.listdir(os.path.join(Segs_Dir, charDir)):
        img = Image.open(os.path.join(Segs_Dir, charDir, file))
        featureVector = np.array(img).flatten()
        Xlist.append(featureVector)
        Ylist.append(charDir)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xlist, Ylist, test_size=0.2)

clf = KNeighborsClassifier(n_jobs=4)
clf.fit(Xtrain, Ytrain)

for n in list(range(1,15)) + list(range(15,100,5)):
    start_t = time.time()
    clf.n_neighbors = n
    Ypredict = clf.predict(Xtest)
    accuracy = accuracy_score(Ytest, Ypredict)
    end_t = time.time()
    ms_per_sample = (end_t-start_t)/len(Xtest) * 1000
    print("{:3d}\t{:.4f}\t{:.3f} ms".format(n, accuracy, ms_per_sample))

