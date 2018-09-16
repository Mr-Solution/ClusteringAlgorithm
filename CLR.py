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

def CLR(A0, c, isrobust=0, islocal=1):
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
    a[a<zr] = sys.float_info.min    # python3 中最小浮点数。 类似的特殊数字还有 sys.maxsize sys.float_info.max float("inf")
    ad = np.diff(a, axis=0)
    ad1 = ad/a[2:]
    ad1[ad1 > 0.85] = 1
    ad1 = ad1 + sys.float_info.min * np.array(range(num-1))
    ad1[0] = 0
    ad1 = ad1[ : math.floor(0.9*len(ad1))]
    te = -np.sort(-ad1)    # 降序排列
    cs = np.argsort(-ad1)  # 降序排列
    print("Suggested cluster number is:",cs[0:5])

    c = cs[1]

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
            idxa0 = np.where(a0 > 0)
        else:
            idxa0 = np.array(range(num))

        u[i] = np.ones((1, len(idxa0)))

    for j in range(NITER):
        dist = distance.cdist(F, F)
        S = np.zeros((num, num))
        for k in range(num):
            a0 = A0[i, :]
            if islocal == 1:
                idxa0 = np.where(a0>0)
            else:
                idxa0 = np.array(range(num))

            ai = a0[idxa0]
            """
            暂停。matlab 代码 CLR.m line：84
            """



        









def eig1(A, c, isMax=1, isSym=1):
    if c > A.shape[0]:
        c = A.shape[0]

    if isSym == 1:
        A = np.maximum(A, A.T)

    d,v = np.linalg.eigh(A)


    d1 = np.sort(d)
    idx = np.argsort(d)
    if isMax == 1:
        d1 = d1[ : :-1]
        idx = idx[ : :-1]

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:,idx1]

    eigval_full = d[idx]

    return eigvec, eigval, eigval_full
