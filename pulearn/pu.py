import numpy as np 

def bagging(mod, X, Y, iters = 100, r = 0.1):
    '''
    Function performs PU bagging
    '''
    p, u = [], []
    for i, y in enumerate(Y):
        if y == 1:
            p.append(i)
        else:
            u.append(i)

    k = np.int(len(u) * r)

    proba = (
        np
        .array([])
        .reshape((0, len(X)))
    )

    for i in range(iters):
        u_perm = list(np.random.permutation(u))
        idx = u_perm[:k] + p
        probs = (
            mod
            .fit(X[idx], Y[idx])
            .predict_proba(X)[:,1]
        )
        proba = np.vstack((proba, probs))

    proba = proba.mean(axis = 0)

    return(proba)


def induction(mod, X, Y, rel_neg = [], iters = 100):
    '''
    Function performs PU inductions (aka. 2-step method)
    '''
    p, u = [], []
    for i, y in enumerate(Y):
        if y == 1:
            p.append(i)
        else:
            u.append(i)

    n = rel_neg
    for i in range(iters):
        proba = (
            mod
            .fit(X[p + n], Y[p + n])
            .predict_proba(X)[:,1]
        )        
        n = [j for j in u if proba[j] < .5]
    
    return proba