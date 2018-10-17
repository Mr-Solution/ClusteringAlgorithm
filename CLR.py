# -*- coding:utf-8 -*-

"""
CLR Algorithm:
  The Constrained Laplacian Rank Algorithm for Graph-Based Clustering
"""

import sys
import math
import numpy as np
import scipy.sparse
from scipy.spatial import distance


def CLR(A0, c=-1, isrobust=0, islocal=1):
    """
    :param A0: affinity matrix
    :param c: cluster number
    :param isrobust:
    :param islocal:
    :return: clustering result, eignvalues of learned graph Laplacian, suggested cluster numbers
    """
    NITER = 30
    zr = 10e-11
    Lambda = 0.1
    r = 0

    A0 = A0 - np.diag(np.diag(A0))
    num = A0.shape[0]
    A10 = (A0 + A0.T)/2
    D10 = np.diag(np.sum(A10, axis=0))
    L0 = D10 - A10
    """
    eig1
    """
    F0, xxxxx, evs = eig1(L0, num, 0)
    a = abs(evs)
    # a[a < zr] = sys.float_info.min  # python3 中最小浮点数。 类似的特殊数字还有 sys.maxsize sys.float_info.max float("inf")
    a[a < zr] = np.spacing(1)    # 对应 matlab 中的 eps. Return the distance between x and the nearest adjacent number
    ad = np.diff(a, axis=0)
    ad1 = ad/a[1:]
    ad1[ad1 > 0.85] = 1
    ad1 = ad1 + sys.float_info.min * np.array(range(num-1))
    ad1[0] = 0
    ad1 = ad1[ : math.floor(0.9*len(ad1))]
    # te = -np.sort(-ad1)    # 降序排列
    cs = np.argsort(-ad1) + 1  # 降序排列
    print("Suggested cluster number is:",cs[0:5])

    if c == -1:
        c = cs[0]

    F = F0[:, :c]

    if np.sum(evs[:c+1]) < zr:
        print("The original graph has more than %d connected component" % c)
        return

    if np.sum(evs[:c]) < zr:
        clusternum, y = scipy.sparse.csgraph.connected_components(scipy.sparse.coo_matrix(A10), connection='strong')
    #    y = y.T
        S = A0
        return

    u = {}
    for i in range(num):
        a0 = A0[i, :]
        if islocal == 1:
            idxa0 = np.where(a0 > 0)[0]
        else:
            idxa0 = np.array(range(num))

        u[i] = np.ones((1, len(idxa0)))

    for j in range(NITER):
        dist = distance.cdist(F, F)
        S = np.zeros((num, num))
        for k in range(num):
            a0 = A0[k, :]
            if islocal == 1:
                idxa0 = np.where(a0 > 0)[0]
            else:
                idxa0 = np.array(range(num))

            ai = a0[idxa0]
            di = dist[k, idxa0]
            if isrobust == 1:
                print("to be continued")
                pass
            else:
                ad = ai - 0.5*Lambda*di
                S[k, idxa0] = EProjSimplex_new(ad)

        A = S
        A = (A + A.T)/2
        D = np.diag(sum(A))
        L = D - A
        F_old = F
        F, idle, ev = eig1(L, c, 0)
        # evs[:,j+1] = ev
        evs = np.vstack((evs, ev))

        fn1 = sum(ev[:c])
        fn2 = sum(ev[:c+1])
        if fn1 > zr:
            Lambda = 2*Lambda
        elif fn2 < zr:
            Lambda = Lambda/2
            F = F_old
        else:
            break

        print("NITER count:", j)

    clusternum, y = scipy.sparse.csgraph.connected_components(scipy.sparse.coo_matrix(A))
    if clusternum != c:
        print("Can not find the correct cluster number")

    return y, evs, cs
    # return y,S,evs,cs


def eig1(A, c, isMax=1, isSym=1):
    if c > A.shape[0]:
        c = A.shape[0]

    if isSym == 1:
        A = np.maximum(A, A.T)

    d, v = np.linalg.eig(A)

    # d1 = np.sort(d)
    # idx = np.argsort(d)
    if isMax == 0:
        # d1 = np.sort(d, kind='mergesort')
        idx = np.argsort(d, kind='mergesort')
        # d1 = d1[ : :-1]
        # idx = idx[ : :-1]
    else:
        # d1 = np.sort(d, kind='mergesort')
        # d1 = d1[:: -1]
        idx = np.argsort(-d, kind='mergesort')

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:,idx1]

    eigval_full = d[idx]

    return eigvec, eigval, eigval_full


def EProjSimplex_new(v, k=1):
    ft = 1
    n = len(v)

    v0 = v - np.mean(v) + k/n
    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            # posidx = v1>0
            npos = np.sum(v1 > 0)
            g = -npos
            f = np.sum(v1[v1 > 0]) - k
            lambda_m = lambda_m - f/g
            ft += 1
            if ft > 100:
                x = np.where(v1 > 0, v1, 0)
                break

        x = np.where(v1 > 0, v1, 0)

    else:
        x = v0

    return x
