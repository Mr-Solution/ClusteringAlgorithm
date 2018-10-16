# -*- coding:utf-8 -*-

import numpy as np
from tool import tool, measure, loadData
import time

if __name__ == '__main__':
    print("hello")
    """ load data """
    # dataset = 'dataset/COIL20_32.txt'
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/USPS.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/TOX.txt'    # K=5, v=1.2, u=1
    # dataset = 'dataset/Jaffe.txt'
    dataset = 'dataset/lung.txt'

    # fea, labels = loadData.load_coil100()

    data = np.loadtxt(dataset)  # K = 25 v = 10 z =0.01
    print("KROD PIC    dataset =", dataset)
    print("data.shape =",data.shape)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)
    """ clustering """
    u = 1
    dist = tool.rank_dis_c(fea, u)
    dist = dist - np.diag(np.diag(dist))

    K = 15
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

    start = time.time()
    cl = tool.gacMerging(graphW, initialCluster, groupNumber, 'path', z)
    end = time.time()
    print("time =",end - start)

    nmi = measure.NMI(labels, cl)
    print("nmi =",nmi)
    acc = measure.ACC(labels, cl)
    print("acc =",acc)

    print("world")





