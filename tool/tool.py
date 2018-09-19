# -*- coding:utf-8 -*-

"""
some functions
"""
import numpy as np
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
        amax = np.max(data[:,i])
        amin = np.min(data[:,i])
        for j in range(m):
            data[j, i] = (data[j,i] - amin)/(amax - amin)

        b1.append(amin)
        b2.append(amax - amin)

    data = np.where(np.isnan(data), 0, data)
    return data #, b1, b2

def rank_dis_c(fea, a):
    dist = distance.cdist(fea, fea)
    dist = dist/np.max(dist)
    list = np.argsort(dist)
    rows = np.argsort(list)
    rodist = rows + rows.T + 2
    rodist = rodist/np.max(rodist)
    rodist = (rodist)*np.exp((dist*dist)/a)
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
def gacBuildDigraph(distance_matrix, K ,a):
    # NN indices
    N = distance_matrix.shape[0]
    sortedDist = np.sort(distance_matrix)
    NNIndex = np.argsort(distance_matrix)
    NNIndex = NNIndex[:, :K+1]
    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K+1,4)], 0))
    tmpNNdist = np.min(sortedDist[:,1:], axis=1)

    while np.any(np.exp(-tmpNNdist/sig2) < 1e-5):
        sig2 = 2*sig2

    ND = sortedDist[:,1:K+1]
    NI = NNIndex[:,1:K+1]
    XI = np.tile(np.arange(N).reshape(N,-1),(1,K))
    graphW = coo_matrix((np.exp(-ND.reshape(-1,order='F')*(1/sig2)), (XI.reshape(-1,order='F'), NI.reshape(-1,order='F'))), shape=(N, N)).todense()
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
        idx = NNIndex[i,:2]
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
                clusterLabels = np.where(clusterLabels==assignedCluster[j], assignedCluster[0], clusterLabels)

    uniqueLabels = np.unique(clusterLabels)



