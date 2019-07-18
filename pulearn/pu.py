import numpy as np 

def bagging(mod, X, Y, n = 100, r = 0.1):
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
    proba = np.array([])

    for i in range(n):
        u_perm = list(np.random.permutation(u))
        train_idx = u_perm[:k] + p
        mod.fit(X[train_idx], Y[train_idx])
        proba.append(mod.predict_prova(X)[:,1])

    proba = proba.mean(axis = 1)

    return(proba)




