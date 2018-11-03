# -*- coding:utf-8 -*-
from tool import tool
import numpy as np
from scipy.sparse import coo_matrix
import copy
import gc


# gdl algorithm
def gdl(dist_set, groupNumber, k=20, a=1, usingKcCluster=True, p=1):
    print("------ Building graph and forming inital clusters with l-links ------")
    graphW, NNIndex = gacBuildDigraph_c(dist_set, k, a)
    initialClusters = gacBuildLlinks_cwarpper(dist_set, p, NNIndex)
    del dist_set, NNIndex
    gc.collect()
    # if usingKcCluster=True    else meixie
    clusteredLabels = gdlMergingKNN_c(graphW, initialClusters, groupNumber)
    return clusteredLabels


# construct a neighborhood graph (directed graph)
def gacBuildDigraph_c(dist_matrix, K, a):
    """
    graph W = [wij],    wij = exp(-dist(i,j)**2/sigma**2), if i is in i's K-nearest neighbors
    :param dist_matrix:
    :param K: K-nearest neighbors
    :param a: free parameter to calculate sigma
    :return:
    """
    N = dist_matrix.shape[0]
    sortedDist, NNIndex = gacMink(dist_matrix, max(K+1, 4), dim=2, axis=1)
    # sig2 = np.mean(sortedDist[:, 1:max(K+1, 4)] ** 2) * a / (N * K)
    sig2 = np.mean(sortedDist[:, 1:max(K + 1, 4)]) * a
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


# initial clusters
def gacBuildLlinks_cwarpper(dist_matrix, p, NNIndex=None):
    """
    The initial small clusters are simply constructed as weakly connected components of a K0-NN graph
    where the neighborhood size K0 is small typically as 1 or 2
    :param dist_matrix:
    :param p:
    :param NNIndex:
    :return:
    """
    if NNIndex is not None:
        NNIndex = NNIndex[:, :p+1]
    else:
        palceholder, NNIndex = gacMink(dist_matrix, p+1, dim=2, axis=1)
    outputClusters = gacOnelink_c(NNIndex)
    return outputClusters


def gacOnelink_c(NNIndex):
    Dim = NNIndex.shape[1]
    N = NNIndex.shape[0]
    visited = -np.ones(N)
    count = 0
    for i in range(N):
        if i==1275:
            x = 1
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
    for i in range(count):
        initialClusters.append(np.where(visited == i)[0].tolist())

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
    # VERBOSE = False
    numClusters = len(initialClusters)
    if numClusters < groupNumber:
        print("err! Too few initial clusters. Do not need merging!")
        return -1
    affinityTab, AsymAffTab = gdlInitAffinityTable_knn_c(graphW, initialClusters, Kc)
    affinityTab = -affinityTab
    affinityTab = affinityTab - np.diag(np.diag(affinityTab)) + np.diag(myInf*np.ones(numClusters))
    placeholder, KcCluster = gacMink(affinityTab, Kc)
    # KcCluster = KcCluster.T

    curGroupNum = numClusters
    while True:
        usingKcCluster = curGroupNum > 1.2*Kc
        minIndex1, minIndex2 = gacPartialMin_knn_c(affinityTab, curGroupNum, KcCluster)
        # print("minIndex1 = %d  minIndex2 = %d" % (minIndex1, minIndex2))

        cluster1 = list(initialClusters[minIndex1])
        cluster2 = list(initialClusters[minIndex2])
        # merge the two clusters
        new_cluster = np.unique(cluster1+cluster2)
        # find candidates to be updated
        if usingKcCluster:
            KcCluster = np.where(KcCluster == minIndex2, minIndex1, KcCluster)
            candidates = np.any(KcCluster == minIndex1, axis=0)
            candidates[np.append(KcCluster[:, minIndex1], KcCluster[:, minIndex2])] = True
            candidates[minIndex1] = False
            candidates[minIndex2] = False
            candidates = np.where(candidates != 0)[0]

        if minIndex2 != curGroupNum:
            initialClusters[minIndex2] = initialClusters[-1]
            affinityTab[:curGroupNum-1, minIndex2] = affinityTab[:curGroupNum-1, curGroupNum-1]
            affinityTab[minIndex2, :curGroupNum-1] = affinityTab[curGroupNum-1, :curGroupNum-1]
            if usingKcCluster:
                KcCluster[:, minIndex2] = KcCluster[:, -1]
                KcCluster = np.where(KcCluster == curGroupNum-1, minIndex2, KcCluster)
                candidates = np.where(candidates == curGroupNum-1, minIndex2, candidates)

        AsymAffTab[:curGroupNum, minIndex1] = AsymAffTab[:curGroupNum, minIndex1] + AsymAffTab[:curGroupNum, minIndex2]
        AsymAffTab[:curGroupNum, minIndex2] = AsymAffTab[:curGroupNum, curGroupNum - 1]
        AsymAffTab[minIndex2, :curGroupNum] = AsymAffTab[curGroupNum-1, :curGroupNum]

        # update the first cluster and remove the second cluster
        initialClusters[minIndex1] = new_cluster
        initialClusters.pop(-1)
        affinityTab[:, curGroupNum-1] = myInf
        affinityTab[curGroupNum-1, :] = myInf
        if usingKcCluster:
            KcCluster = np.delete(KcCluster, -1, axis=1)
        curGroupNum = curGroupNum - 1
        if curGroupNum <= groupNumber:
            break

        # if usingKcCluster and minIndex2 != curGroupNum:
        #     candidates = np.where(candidates == curGroupNum, minIndex2, candidates)

        # update the affinity table for the merged cluster
        if usingKcCluster:
            affinityTab[:curGroupNum, minIndex1] = myInf
            for groupIndex in candidates:
                if AsymAffTab[minIndex1, groupIndex] > -myBoundInf and AsymAffTab[groupIndex, minIndex1] > -myBoundInf:
                    AsymAffTab[minIndex1, groupIndex] = gdlDirectedAffinity_c(graphW, initialClusters, minIndex1, groupIndex)
                else:
                    AsymAffTab[groupIndex, minIndex1], AsymAffTab[minIndex1, groupIndex] = gdlAffinity_c(graphW, initialClusters[groupIndex], new_cluster)
            affinityTab[candidates, minIndex1] = -AsymAffTab[minIndex1, candidates].T - AsymAffTab[candidates, minIndex1]
        else:
            affinityTab[minIndex1, minIndex1] = myInf
            for groupIndex in range(curGroupNum):
                if groupIndex == minIndex1:
                    continue
                if AsymAffTab[minIndex1, groupIndex] > -myBoundInf and AsymAffTab[groupIndex, minIndex1] > -myBoundInf:
                    AsymAffTab[minIndex1, groupIndex] = gdlDirectedAffinity_c(graphW, initialClusters, minIndex1, groupIndex)
                else:
                    AsymAffTab[groupIndex, minIndex1], AsymAffTab[minIndex1, groupIndex] = gdlAffinity_c(graphW, initialClusters[groupIndex], new_cluster)
            affinityTab[:curGroupNum, minIndex1] = -AsymAffTab[minIndex1, :curGroupNum].T - AsymAffTab[:curGroupNum, minIndex1]

        affinityTab[minIndex1, :curGroupNum] = affinityTab[:curGroupNum, minIndex1].T
        if usingKcCluster:
            placeholder, KcCluster[:, minIndex1] = gacMink(affinityTab[:curGroupNum, minIndex1], Kc, 1)

    # generate sample labels
    clusterLabels = np.ones(numSample)
    for i in range(len(initialClusters)):
        clusterLabels[initialClusters[i]] = i

    return clusterLabels


# calculate the affinity between two clusters
def gdlAffinity_c(graphW, cluster_i, cluster_j):
    num_i = len(cluster_i)
    num_j = len(cluster_j)
    sum1 = 0
    # affinity between vertexs in cluster_i and cluster_j
    # for j in range(num_j):
    #     index_j = cluster_j[j]
    #     Lij = 0
    #     Lji = 0
    #     for i in range(num_i):
    #         index_i = cluster_i[i]
    #         Lij += graphW[index_i, index_j]
    #         Lji += graphW[index_j, index_i]
    #     sum1 += Lij * Lji

    for j in cluster_j:
        indegree = 0    # indegree
        outdegree = 0    # outdegree
        for i in cluster_i:
            indegree += graphW[i, j]
            outdegree += graphW[j, i]
        sum1 += indegree * outdegree

    # affinity between vertex in cluster_j and cluster_i
    sum2 = 0
    # for i in range(num_i):
    #     index_i = cluster_i[i]
    #     Lij = 0
    #     Lji = 0
    #     for j in range(num_j):
    #         index_j = cluster_j[j]
    #         Lji += graphW[index_j, index_i]
    #         Lij += graphW[index_i, index_j]
    #     sum2 += Lji * Lij

    for i in cluster_i:
        indegree = 0
        outdegree = 0
        for j in cluster_j:
            indegree += graphW[j, i]
            outdegree  += graphW[i, j]
        sum2 += indegree * outdegree

    return sum1/(num_i*num_i), sum2/(num_j*num_j)


def gdlDirectedAffinity_c(graphW, initialClusters, i, j):
    cluster_i = list(initialClusters[i])
    cluster_j = list(initialClusters[j])
    num_i = len(cluster_i)
    num_j = len(cluster_j)
    sum = 0
    for j in range(num_j):
        index_j = cluster_j[j]
        Lij = 0
        Lji = 0
        for i in range(num_i):
            index_i = cluster_i[i]
            Lij += graphW[index_i, index_j]
            Lji += graphW[index_j, index_i]
        sum += Lij*Lji
    return sum/(num_i*num_i)


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
        for j in range(curGroupNum):
            for i in range(Kc):
                index_i = KcCluster[i, j]
                if 0 <= index_i < curGroupNum and affinityTab[index_i, j] < minElem:
                    minElem = affinityTab[index_i, j]
                    minIndex1 = index_i
                    minIndex2 = j
    else:
        # KcCluster2 = copy.deepcopy(KcCluster)
        # np.where(KcCluster2 < curGroupNum, KcCluster2, curGroupNum)
        # minIndex1, minIndex2 = np.where(affinityTab == np.min(affinityTab[np.unique(KcCluster), :])) # error

        # for j in range(curGroupNum):
        #     for i in range(Kc):
        #         index_i = KcCluster[i, j]
        #         if 0 <= index_i < curGroupNum:
        #             if affinityTab[index_i, j] < minElem:
        #                 minElem = affinityTab[index_i, j]
        #                 minIndex1 = index_i
        #                 minIndex2 = j

        xids = KcCluster.T
        yids = np.arange(curGroupNum)
        xids = xids.reshape((1, -1))
        yids = np.tile(yids, (Kc, 1))
        yids = yids.T
        yids = yids.reshape((1, -1))
        minElem = np.min(affinityTab[xids, yids])
        minIndex1, minIndex2 = np.where(affinityTab == minElem)
        minIndex1 = minIndex1[-1]
        minIndex2 = minIndex2[-1]

    if minIndex1 > minIndex2:
        minIndex1, minIndex2 = minIndex2, minIndex1

    # print("minIndex1 = %d, minIndex2 = %d" % (minIndex1, minIndex2))

    return minIndex1, minIndex2


def gdlInitAffinityTable_knn_c(graphW, initClusters, Kc):
    numClusters = len(initClusters)
    affinityTab = np.zeros((numClusters, numClusters))  #-1e10 * np.ones((numClusters, numClusters))
    # AsymAffTab = np.zeros((numClusters, numClusters))  #-1e10 * np.ones((numClusters, numClusters))
    # asymmetric affinity from cluster_i to cluster_j
    AsymAffTab = -1e10 * np.ones((numClusters, numClusters))

    for j in range(numClusters):
        cluster_j = initClusters[j]
        for i in range(j):
            cluster_i = initClusters[i]
            affinityTab[i, j] = -computeAverageDegreeAffinity(graphW, cluster_i, cluster_j)
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
            if inKcCluster[i, j]:
                tmpAsymAff0, tmpAsymAff1 = gdlComputeAffinity(graphW, initClusters[i], cluster_j)
                affinityTab[i, j] = tmpAsymAff0 + tmpAsymAff1
                AsymAffTab[i, j] = tmpAsymAff0
                AsymAffTab[j, i] = tmpAsymAff1
            else:
                affinityTab[i, j] = -1e10

    # AsymAffTab = AsymAffTab + np.diag(-1e10 * np.ones(numClusters))
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


# the product of average indegree and average outdegree
def computeAverageDegreeAffinity(graphW, cluster_i, cluster_j):
    sum = 0
    for i in cluster_i:
        for j in cluster_j:
            sum += graphW[i, j] + graphW[j, i]
    return sum/(len(cluster_i) * len(cluster_j))


def gacFindKcCluster(affinityTab, Kc):
    Kc = np.ceil(1.2*Kc).astype(np.int)
    sortedAff, placeholder = gacMink(affinityTab, Kc, dim=2, axis=0)
    # inKcCluster = affinityTab <= sortedAff[:, Kc-1]
    inKcCluster = affinityTab <= np.tile(sortedAff[Kc-1, :], (affinityTab.shape[0], 1))
    inKcCluster = inKcCluster | inKcCluster.T
    return inKcCluster


# C++ std::partial_sort
def gacMink(X, k, dim=2, axis=0):
    # sortedDist, NNIndex = gacPartial_sort(X, k, dim)
    if dim == 2:
        if axis == 0:    # 按列排
            sortedDist = np.sort(X, kind='mergesort', axis=0)[:k, :]
            NNIndex = np.argsort(X, kind='mergesort', axis=0)[:k, :]
        else:    # 按行排
            sortedDist = np.sort(X, kind='mergesort', axis=1)[:, :k]
            NNIndex = np.argsort(X, kind='mergesort', axis=1)[:, :k]
    if dim == 1:
        sortedDist = np.sort(X, kind='mergesort')[:k]
        NNIndex = np.argsort(X, kind='mergesort')[:k]

    return sortedDist, NNIndex


# def gacPartial_sort(X, k, dim):
#     sortedDist = np.sort(X, axis=1)
#     NNIndex = np.argsort(X, axis=1)
#     return sortedDist, NNIndex
