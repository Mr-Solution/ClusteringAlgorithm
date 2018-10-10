# -*- coding:utf-8 -*-

import numpy as np
from tool import tool,measure
import time

if __name__ == '__main__':
    print("hello KROD PIC")
    """ load data """
    # data = np.loadtxt('dataset/COIL20_32.txt')
    data = np.loadtxt('dataset/mnist.txt')  # K = 25 v = 10 z =0.01
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)
    """ clustering """
    u = 1
    dist = tool.rank_dis_c(fea, u)
    dist = dist - np.diag(np.diag(dist))

    K = 25
    v = 10
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

    start = time.time()
    cl = tool.gacMerging(graphW, initialCluster, groupNumber, 'path', z)
    end = time.time()
    print("time =",end - start)

    nmi = measure.NMI(labels, cl)
    print("nmi =",nmi)
    acc = measure.ACC(labels, cl)
    print("acc =",acc)

    print("world")





