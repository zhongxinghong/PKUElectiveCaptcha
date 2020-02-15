from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
from PIL import Image
np.random.seed(0)
path = 'segs/'
Xlist = []
Ylist = []
for directory in os.listdir(path):
    for file in os.listdir(path + directory):
        print(path + directory + "/" + file)
        img = Image.open(path + directory + "/" + file)
        featurevector = np.array(img).flatten()
        Xlist.append(featurevector)
        Ylist.append(directory)

Xtrain, Xtest, ytrain, ytest = train_test_split(Xlist, Ylist, test_size=0.1)

clf = KNeighborsClassifier()

clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)
accuracy = accuracy_score(ytest, ypred)
print("acc：", accuracy)

import time
start = time.time()
ypred = clf.predict([Xtest[0]])
end = time.time()

print('time per item: {}'.format(end-start))
