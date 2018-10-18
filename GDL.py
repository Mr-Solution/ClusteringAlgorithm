# -*- coding:utf-8 -*-

from tool import tool
import numpy as np
from scipy.sparse import coo_matrix


def gdl(dist_set, groupNumber, K=20, a=1, usingKcCluster=True, p=1):
    print("------ Building graph and forming iniital clusters with l-links ------")
    graphW, NNIndex = gacBuildDigraph_c(dist_set, K, a)
    initialClusters = gacBuildLlinks_cwarpper(dist_set, p, NNIndex)
    # if usingKcCluster=True    else meixie
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
    palceholder, NNIndex = gacMink(dist_matrix, p+1, 2)
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


# Cluster merging for Graph Degree Linkage
def gdlMergingKNN_c(graphW, initialClusters, groupNumber):
    """
    :param graphW: asymmetric weighted adjacency matrix
    :param initialClusters:  a cell array of clustered vertices
    :param groupNumber: the final number of clusters
    :return: clusterlabels 1xm array
    """
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
    curGroupNum = numClusters
    while True:
        usingKcCluster = curGroupNum > 1.2*Kc
        minIndex1, minIndex2 = gacPartialMin_knn_c(affinityTab, curGroupNum, KcCluster)
        cluster1 = initialClusters[minIndex1]
        cluster2 = initialClusters[minIndex2]


def gacPartialMin_knn_c(affinityTab, curGroupNum, KcCluster):
    """
    :param affinityTab: matrix
    :param curGroupNum: int
    :param KcCluster: matrix
    :return:
    """
    numClusters = affinityTab.shape[0]
    Kc = KcCluster.shape[0]
    minIndex1 = 0
    minIndex2 = 0
    minElem = 1e10
    if curGroupNum < 1.2*Kc:
        

    return minIndex1, minIndex2


def gdlInitAffinityTable_knn_c(graphW, initClusters, Kc):
    numClusters = len(initClusters)
    affinityTab = np.zeros((numClusters, numClusters))  #-1e10 * np.ones((numClusters, numClusters))
    AsymAffTab = np.zeros((numClusters, numClusters))  #-1e10 * np.ones((numClusters, numClusters))

    for j in range(numClusters):
        cluster_j = initClusters[j]
        for i in range(j):
            cluster_i = initClusters[i]
            affinityTab[i,j] = -computeAverageDegreeAffinity(graphW, cluster_i, cluster_j)
        #affinityTab[j,j] = -1e10

    # from upper triangular to full symmetric
    affinityTab += affinityTab.T
    affinityTab += np.diag(-1e10*np.ones(numClusters))

    # sort
    inKcCluster = gacFindKcCluster(affinityTab, Kc)

    # computing
    for j in range(numClusters):
        cluster_j = initClusters[j]
        for i in range(j):
            if inKcCluster[i,j]:
                tmpAsymAff0, tmpAsymAff1 = gdlComputeAffinity(graphW, initClusters[i], cluster_j)
                affinityTab[i,j] = tmpAsymAff0 + tmpAsymAff1
                AsymAffTab[i,j] = tmpAsymAff0
                AsymAffTab[j,i] = tmpAsymAff1
            else:
                affinityTab[i,j] = -1e10

    # from upper triangular to full symmetric
    affinityTab = np.triu(affinityTab, 1) + np.triu(affinityTab, 1).T + np.diag(np.diag(affinityTab))
    # affinityTab += np.triu(affinityTab, 1).T

    return affinityTab, AsymAffTab


# 这个函数不对
def gdlComputeAffinity(pW, cluster_i, cluster_j):
    num_i = len(cluster_i)
    num_j = len(cluster_j)
    sum1 = 0
    for j in cluster_j:
        Lij = 0
        Lji = 0
        for i in cluster_i:
            Lij += pW[i, j]
            Lji += pW[j, i]
        sum1 += Lij * Lji

    sum2 = 0
    for i in cluster_i:
        Lij = 0
        Lji = 0
        for j in cluster_j:
            Lji += pW[j, i]
            Lij += pW[i, j]
        sum2 += Lji * Lij

    return sum1/(num_i*num_i), sum2/(num_j*num_j)


def computeAverageDegreeAffinity(graphW, cluster_i, cluster_j):
    sum = 0
    for i in cluster_i:
        for j in cluster_j:
            sum += graphW[i,j] + graphW[j,i]
    return sum/(len(cluster_i) * len(cluster_j))


def gacFindKcCluster(affinityTab, Kc):
    Kc = np.ceil(1.2*Kc).astype(np.int)
    sortedAff, placeholder = gacMink(affinityTab, Kc, 1)
    inKcCluster = affinityTab <= sortedAff[:,Kc-1].T
    inKcCluster = inKcCluster | inKcCluster.T
    return inKcCluster


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
