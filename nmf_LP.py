# -*- coding:utf-8 -*-
"""
from tool import tool
import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy import sparse

def predict_nmf_lp(fea, nclass):
    NMFKLoptions = {}
    NMFKLoptions['maxIter'] = 50
    # when alpha = 0, GNMF_KL boils down to the ordinary NMF_KL.
    NMFKLoptions['alpha'] = 0
    NMFKLoptions['weight'] = 'NCW'
    nFactor = 10
    placeholder, V = GNMF_KL(fea, nFactor, [], NMFKLoptions)
    kmeans = KMeans(init='k-means++', n_clusters=nclass, n_init=20)
    kmeans.fit(fea)
    return kmeans.labels_

def GNMF_KL(X, k, W, options):
    ZERO_OFFSET = 1e-200
    nSmp = X.shape[0]
    if np.min(X) < 0:
        print("err! Input should be nonnegative")
        return -1
    if 'error' not in options:
        options['error'] = 1e-5
    if 'maxIter' not in options:
        options['maxIter'] = []
    if 'nRepeat' not in options:
        options['nRepeat'] = 10
    if 'minIter' not in options:
        options['minIter'] = 10
    if 'meanFitRatio' not in options:
        options['meanFitRatio'] = 0.1
    if 'alpha' not in options:
        options['alpha'] = 100
    if 'alpha_nSmp' in options and options['alpha_nSmp']:
        options['alpha'] = options['alpha'] * nSmp
    if 'kmeansInit' not in options:
        options['kmeansInit'] = 1

    NCWeight = []
    if 'weight' in options and options['weight'] == "NCW":
        feaSum = np.sum(X, axis=0)
        NCWeight = 1/np.dot(X, feaSum)
        tmpNCWeight = NCWeight
    else:
        tmpNCWeight = np.ones((nSmp,1))

    if sparse.issparse(X):
        # matlab nnz(X)返回矩阵X中的非零元素的数目
        nz = len(np.nonzero(X)[0])
        nzk = nz*k
        idx, jdx = np.nonzero(X)
        vdx = X[X!=0]
        if not NCWeight:    # isempty(NCWeight)
            Cons = sum(vdx * np.log(vdx) - vdx)
        else:
            Cons = sum(NCWeight[jdx] * (vdx * np.log(vdx) - vdx))

        # ldx 暂时没必要
        options['nz'] = nz
        options['nzk'] = nzk
        options['idx'] = idx
        options['jdx'] = jdx
        options['vdx'] = vdx

    else:
        Y = X + ZERO_OFFSET
        if not NCWeight:
            Cons = np.sum(Y * np.log(Y) - Y)
        else:
            Cons = np.sum(NCWeight * np.sum(Y*np.log(Y)-Y, axis=0))

    options['NCWeight'] = NCWeight
    options['tmpNCWeight'] = tmpNCWeight
    options['Cons'] = Cons

    if 'Optimization' not in options:
        options['Optimization'] = "Multiplicative"

    U_final, V_final, nIter_final, objhistory_final = GNMF_KL_Multi(X, k, W, options)

    return U_final, V_final, nIter_final, objhistory_final


def GNMF_KL_Multi(X, k, W, options, U, V):
    differror = options['error'];
    maxIter = options['maxIter'];
    nRepeat = options['nRepeat'];
    minIter = options['minIter'] - 1;
    if maxIter and maxIter < minIter+1:
        minIter = maxIter-1
    meanFitRatio = options['meanFitRation']
    kmeansInit = options['kmeansInit']
    alpha = options['alpha']
    NCWeight = options['NCWeight']
    tmpNCWeight = options['tmpNCWeight']
    Cons = options['Cons']
    if sparse.issparse(X):
        nz = options['nz'];
        nzk = options['nzk'];
        idx = options['idx'];
        jdx = options['jdx'];
        vdx = options['vdx'];
        #ldx = options.ldx;
    Norm = 2
    NormV = 0
    mFea, nSmp = X.shape

    if alpha > 0:
        # oridinary NMF_KL alpha = 0
        pass
    else:
        L = []

    selectInit = 1
    if not U:
        if kmeansInit:
            U,V = NMF_init(X,k)
        else:
            U = abs(np.random.rand(mFea,k))
            V = abs(np.random.rand(nSmp,k))
    else:
        nRepeat = 1




    return U_final, V_final, nIter_final, objhistory_final



def NMF_init(X, k):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(X)
    # cl = kmeans.labels_
    center = kmeans.cluster_centers_
    center = np.where(center>0, center, 0)
    U = center.T
    UTU = U.T*U
    UTU = np.where(UTU > UTU.T, UTU, UTU.T)
    UTX = center*X

    V = np.where(np.dot(np.linalg.inv(UTU), UTX) > 0, np.dot(np.linalg.inv(UTU), UTX), 0)
    return U, V.T

def NormalizeUV(U, V, NormV, Norm):
    K = U.shape[1]
    if Norm == 2:
        if NormV:
            norms = np.where(np.sqrt())

"""