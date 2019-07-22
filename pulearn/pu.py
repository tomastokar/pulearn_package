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


