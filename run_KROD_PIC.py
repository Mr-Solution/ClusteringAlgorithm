# -*- coding:utf-8 -*-

import tool.tool as tool
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("hello")
    """ load data """
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)
    """ clustering """
    u = 1
    dist = tool.rank_dis_c(fea, u)
    dist = dist - np.diag(np.diag(dist))

    K = 5
    v = 1
    z = 0.01
    groupNumber = len(np.unique(labels))
    ND = dist.shape[0]

    # build the adjacency graph
    graphW, NNIndex = tool.gacBuildDigraph(dist, K, v)
    graphW = np.around(graphW, decimals=4)
    # from adjacency matrix to probability transition matrix
    def f(x):
        return x/np.sum(x)
    graphW = np.apply_along_axis(f, 1, graphW)

    initialCluster = tool.gacNNMerge(dist, NNIndex)
    numClusters = len(initialCluster)
    cl = tool.gacMerging(graphW, initialCluster, groupNumber, 'path', z)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print("world")





