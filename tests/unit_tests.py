import numpy as np
from sklearn import datasets, svm
from pulearn.pu import bagging, induction


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


def test_pu_induction():
    '''
    Function will test pu induction
    '''
    p_num = 20
    rn_num = 10
    iris = datasets.load_iris()
    rn = np.where(iris.target == 2)[0]
    rn = np.random.choice(rn, rn_num)
    X = iris.data[:,:2]
    Y = np.array([1] * p_num  + [0] * (len(X) - p_num))
    mod = svm.SVC(gamma='scale', probability=True)
    pu_proba = induction(mod, X, Y, rel_neg = rn, iters = 1000)
    assert type(pu_proba) == np.ndarray
    assert len(pu_proba) == len(X)
    assert np.min(pu_proba) < 1.
    assert np.max(pu_proba) > 0.



