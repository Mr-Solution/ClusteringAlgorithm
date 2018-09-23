# -*- coding:utf-8 -*-

"""
some functions
"""
import numpy as np
import numpy.linalg as nlg
from scipy.spatial import distance
from scipy.sparse import coo_matrix


def data_Normalized(data):
    """
    :param data:
    :return: normalized data
    row: samples
    cul: features
    """
    m, n = data.shape
    b1 = []
    b2 = []
    # 遍历每一个特征
    for i in range(n):
        amax = np.max(data[:, i])
        amin = np.min(data[:, i])
        for j in range(m):
            data[j, i] = (data[j, i] - amin) / (amax - amin)

        b1.append(amin)
        b2.append(amax - amin)

    data = np.where(np.isnan(data), 0, data)
    return data  # , b1, b2


def rank_dis_c(fea, a):
    dist = distance.cdist(fea, fea)
    dist = dist / np.max(dist)
    list = np.argsort(dist)
    rows = np.argsort(list)
    rodist = rows + rows.T + 2
    rodist = rodist / np.max(rodist)
    rodist = (rodist) * np.exp((dist * dist) / a)
    return rodist


"""
build directed graph
arguments:
    distance_matrix: pairwise distances
    K: the number of nearest neighbors for KNN graph
    a: for covariance estimation
return:
    graphW: asymmetric weighted adjacency matrix
    NNIndex: nearest neighbors, N*(2K+1) matrix
"""
def gacBuildDigraph(distance_matrix, K, a):
    # NN indices
    N = distance_matrix.shape[0]
    sortedDist = np.sort(distance_matrix)
    NNIndex = np.argsort(distance_matrix)
    NNIndex = NNIndex[:, :K + 1]
    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K + 1, 4)], 0))
    tmpNNdist = np.min(sortedDist[:, 1:], axis=1)

    while np.any(np.exp(-tmpNNdist / sig2) < 1e-5):
        sig2 = 2 * sig2

    ND = sortedDist[:, 1:K + 1]
    NI = NNIndex[:, 1:K + 1]
    XI = np.tile(np.arange(N).reshape(N, -1), (1, K))
    graphW = coo_matrix(
        (np.exp(-ND.reshape(-1, order='F') * (1 / sig2)), (XI.reshape(-1, order='F'), NI.reshape(-1, order='F'))),
        shape=(N, N)).todense()
    graphW = graphW - np.diag(np.diag(graphW)) + np.eye(N)

    return graphW, NNIndex


"""
merge each vertex with its nearest neighbor
"""
def gacNNMerge(distance_matrix, NNIndex):
    sampleNum = distance_matrix.shape[0]

    clusterLabels = np.zeros(sampleNum)
    counter = 1
    for i in range(sampleNum):
        idx = NNIndex[i, :2]
        assignedCluster = clusterLabels[idx]
        assignedCluster = np.unique(assignedCluster[np.where(assignedCluster > 0)])
        if len(assignedCluster) == 0:
            clusterLabels[idx] = counter
            counter += 1
        elif len(assignedCluster) == 1:
            clusterLabels[idx] = assignedCluster
        else:
            clusterLabels[idx] = assignedCluster[0]
            for j in range(1, len(assignedCluster)):
                clusterLabels = np.where(clusterLabels == assignedCluster[j], assignedCluster[0], clusterLabels)

    uniqueLabels = np.unique(clusterLabels)
    initialClusters = []  # 二维 list 似乎比 dict 更接近 matlab 中的 cell
    for i in range(len(uniqueLabels)):
        initialClusters.append(np.where(clusterLabels == uniqueLabels[i])[0])
        # np.where() 返回一个tuple，tuple[0] 取其第一个元素，即ndarray

    return initialClusters


def gacMerging(graphW, initClusters, groupNumbers, strDescr, z):
    numSample = graphW.shape[0]
    IminuszW = np.eye(numSample) - z * graphW
    myInf = 1e10
    # initialization
    VERBOSE = True

    if strDescr == "zeta":
        complexity_fun = gacZetaEntropy
        conditionalComplexity_fun = gacZetaCondEntropy
    elif strDescr == "path":
        complexity_fun = gacPathEntropy
        conditionalComplexity_fun = gacPathCondEntropy
    else:
        print("ERROR")
        return

    numClusters = len(initClusters)
    if numClusters <= groupNumbers:
        print("err! too few initial clusters. Do not need merging!")

    clusterComp = np.zeros(numClusters)  # ((numClusters,1))
    for i in range(numClusters):
        clusterComp[i] = complexity_fun(IminuszW[initClusters[i],:][:, initClusters[i]])

    if VERBOSE:
        print("Computing initial table.")

    affinityTab = np.full((numClusters, numClusters), np.inf)
    for j in range(numClusters):
        for i in range(j):
            affinityTab[i, j] = -conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])

    # numpy在处理向量加法的时候若维度不对应，会自动转换为矩阵
    affinityTab = \
        affinityTab + np.tile(clusterComp,(len(clusterComp),1)) + np.tile(clusterComp.reshape(-1, 1), (1,len(clusterComp)))
    # matlab 浮点计算，太接近0的数字会被记为0， 这里过滤一遍affinityTab
    affinityTab[abs(affinityTab) < 1e-10] = 0

    if VERBOSE:
        print("Starting merging process")

    curGroupNum = numClusters
    while True:
        if curGroupNum % 20 == 0 and VERBOSE:
            print("Group count:", curGroupNum)

        minAff = np.min(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex2 = np.argmin(minAff)
        minIndex1 = minIndex1[minIndex2]

        if minIndex2 < minIndex1:
            minIndex1, minIndex2 = minIndex2, minIndex1

        new_cluster = np.unique(np.hstack((initClusters[minIndex1], initClusters[minIndex2])))

        if minIndex2 != curGroupNum:
            initClusters[minIndex2] = initClusters[-1]
            clusterComp[minIndex2] = clusterComp[curGroupNum - 1]
            affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
            affinityTab[minIndex2, minIndex2 + 1:curGroupNum] = affinityTab[minIndex2 + 1:curGroupNum, curGroupNum-1]

        # update the first cluster and remove the second cluster
        initClusters[minIndex1] = new_cluster
        initClusters.pop(-1);
        clusterComp[minIndex1] = complexity_fun(IminuszW[new_cluster,:][:, new_cluster])
        clusterComp[curGroupNum-1] = myInf
        affinityTab[:, curGroupNum-1] = myInf
        affinityTab[curGroupNum-1, :] = myInf
        curGroupNum -= 1
        if curGroupNum <= groupNumbers:
            break

        # update the affinity table for the merged cluster
        for groupIndex1 in range(minIndex1 - 1):
            affinityTab[groupIndex1, minIndex1] = \
                -conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)

        for groupIndex1 in range(minIndex1 + 1, curGroupNum):
            affinityTab[minIndex1, groupIndex1] = \
                -conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)

        affinityTab[:minIndex1 - 1, minIndex1] = \
            clusterComp[:minIndex1 - 1] + clusterComp[minIndex1] + affinityTab[:minIndex1 - 1, minIndex1]
        affinityTab[minIndex1, minIndex1+1:curGroupNum] = \
            clusterComp[minIndex1 + 1:curGroupNum] + clusterComp[minIndex1] + affinityTab[minIndex1, minIndex1 + 1:curGroupNum]

    # generate sample labels
    clusterLabels = np.ones(numSample)
    for i in range(len(initClusters)):
        clusterLabels[initClusters[i]] = i

    if VERBOSE:
        print("Final group count:", curGroupNum)

    return  clusterLabels


def gacPathEntropy(subIminuszW):
    N = subIminuszW.shape[0]
    clusterComp = np.dot(nlg.inv(subIminuszW), np.ones((N, 1)))
    clusterComp = np.sum(clusterComp) / (N * N)
    return clusterComp


def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
    num_i = cluster_i.size
    num_j = cluster_j.size

    ijGroupIndex = np.hstack((cluster_i, cluster_j))

    y_ij = np.zeros((num_i + num_j, 2))
    y_ij[:num_i, 0] = 1
    y_ij[num_i:, 1] = 1

    L_ij = np.dot(nlg.inv(IminuszW[ijGroupIndex,:][:, ijGroupIndex]), y_ij)
    L_ij = np.sum(L_ij[:num_i, 0]) / (num_i * num_i) + np.sum(L_ij[num_i:, 1]) / (num_j * num_j)
    return L_ij


def gacZetaEntropy(subIminuszW):
    clusterComp = np.sum(np.log(np.diag(nlg.inv(subIminuszW)).real)) / subIminuszW.shape[0]
    return clusterComp


def gacZetaCondEntropy(IminuszW, cluster_i, cluster_j):
    num_i = len(cluster_i)
    num_j = len(cluster_j)

    ijGroupIndex = np.vstack((num_i, num_j))
    logZetaSelfSim = np.log(np.diag(nlg.inv(IminuszW[ijGroupIndex, ijGroupIndex])).real)
    L_ij = np.sum(logZetaSelfSim[:num_i]) / num_i + np.sum(logZetaSelfSim[num_i:]) / num_j
    return L_ij


"""
construct similarity matrix with probabilistic k-nearest neighbors
It is a parameter free, distance consistent similarity
arguments:
X: each row is a data point
k: number of neighbors
issymmetric: set W = (W+W')/2 if issymmetric=1
return:
W = similarity matrix
"""
def constructW_PKN(X, k=5, issymmetric=1):
    n = X.shape[0]
    dim = X.shape[1]
    D = distance.cdist(X, X)
    idx = np.argsort(D, axis=1)

    W = np.zeros((n,n))
    for i in range(n):
        id = idx[i, 1:k+1]
        di = D[i, id]
        W[i,id] = (di[k]-di)/(k*di[k]-np.sum(di[:k])+np.spacing(1))

    if issymmetric == 1:
        W = (W+W.T)/2

    return W