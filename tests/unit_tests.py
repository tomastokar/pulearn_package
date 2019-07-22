import numpy as np
from sklearn import datasets, svm
from pulearn.pu import bagging


def test_pu_bagging():
    '''
    Function will test pu bagging
    '''
    p_num = 20
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    Y = np.array([1] * p_num  + [0] * (len(X) - p_num))
    mod = svm.SVC(gamma='scale', probability=True)
    pu_proba = bagging(mod, X, Y, iters = 1000, r = .2)
    assert type(pu_proba) == np.ndarray
    assert len(pu_proba) == len(X)
    assert np.min(pu_proba) < 1.
    assert np.max(pu_proba) > 0.

