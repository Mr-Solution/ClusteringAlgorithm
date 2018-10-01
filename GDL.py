# -*- coding:utf-8 -*-

from tool import tool
import numpy as np
from scipy.sparse import coo_matrix

def gdl(dist_set, groupNumber, K=20, a=1, usingKcCluster=True, p=1):
    print("------ Building graph and forming iniital clusters with l-links ------")
    graphW, NNIndex = gacBuildDigraph_c(dist_set, K, a)
    initialClusters = gacBuildLlinks_cwarpper(dist_set, p, NNIndex)
    clusteredLabels = gdlMergingKNN_c(graphW, initialClusters, groupNumber)
    return clusteredLabels

def gacBuildDigraph_c(dist_matrix, K, a):
    N = dist_matrix.shape[0]
    sortedDist, NNIndex = gacMink(dist_matrix, max(K+1, 4), 2)
    sig2 = np.mean(sortedDist[:,1:max(K+1,4)]) * a
    tmpNNDist = np.min(sortedDist[:,1:], axis=1)

    while np.any(np.exp(-tmpNNDist/sig2) < 1e-5):
        sig2 = 2*sig2

    print("sigma =", np.sqrt(sig2))

    ND = sortedDist[:,1:K+1]
    NI = NNIndex[:,1:K+1]
    XI = np.tile(np.arange(N), (K, 1)).T
    graphW = coo_matrix((np.exp(-ND.reshape(-1,order='F')/sig2), (XI.reshape(-1, order='F'), NI.reshape(-1,order='F'))), shape=(N,N)).toarray()
    graphW = graphW + np.eye(N)
    return graphW, NNIndex



def gacBuildLlinks_cwarpper(dist_matrix, p, NNIndex):
    palceholder,NNIndex = gacMink(dist_matrix, p+1, 2)
    outputClusters = gacOnelink_c(NNIndex)
    return outputClusters

def gacOnelink_c(NNIndex):
    Dim = NNIndex.shape[1]
    N = NNIndex.shape[0]
    visited = -np.ones(N)
    count = 0
    for i in range(N):
        linkedIdx = []
        cur_idx = i
        while visited[cur_idx] == -1:
            linkedIdx.append(cur_idx)
            visited[cur_idx] = -2    # -2 is for visited but not assigned
            cur_idx = NNIndex[cur_idx,1]
        if visited[cur_idx] < 0:
            visited[cur_idx] = count
            count += 1
        cluster_id = visited[cur_idx]
        while len(linkedIdx):    # 非空
            visited[linkedIdx.pop()] = cluster_id

    initialClusters = []
    visited = np.array(visited)
    for id in range(count):
        indics = np.where(visited==id)[0].tolist()
        initialClusters.append(indics)

    return initialClusters

def gdlMergingKNN_c(graphW, initialClusters, groupNumber):
    numSample = graphW.shape[0]
    myInf = 1e10
    myBoundInf = 1e8
    Kc = 10
    VERBOSE = False
    numClusters = len(initialClusters)
    if numClusters < groupNumber:
        print("err! Too few initial clusters. Do not need merging!")
        return -1
    affinityTab, AsymAffTab = gdlInitAffinityTable_knn_c(graphW, initialClusters, Kc)
    affinityTab = -affinityTab
    affinityTab = affinityTab - np.diag(np.diag(affinityTab)) + np.diag(myInf*np.ones(numClusters))
    placeholder, KcCluster = gacMink(affinityTab, Kc)
    """
    matlab gdlMergingKNN_c line:41
    """


def gacMink (X, k, dim=1):
    # sortedDist, NNIndex = gacPartial_sort(X, k, dim)
    sortedDist = np.sort(X, axis=1)
    NNIndex = np.argsort(X, axis=1)
    sortedDist = sortedDist[:, :k]
    NNIndex = NNIndex[:, :k]
    return sortedDist, NNIndex

def gacPartial_sort(X, k, dim):
    # nrows, ncols = X.shape
    # partial_sort_rows_withIdx((double *)outdata, pIdx, (double *)indata, rank, ncols, nrows);
    sortedDist = np.sort(X, axis=1)
    NNIndex = np.argsort(X, axis=1)
    return sortedDist, NNIndex