from sklearn import datasets, svm
from pulearn.pu import bagging

import numpy as np

iris = datasets.load_iris()
p_num = 20
X = iris.data[:,:2]
Y = [1] * p_num  + [0] * (len(X) - p_num)

p, u = [], []
for i, y in enumerate(Y):
    if y == 1:
        p.append(i)
    else:
        u.append(i)

r = 0.1
k = np.int(len(u) * r)