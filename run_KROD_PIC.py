# -*- coding:utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tool import tool, measure, loadData
import time

if __name__ == '__main__':
    print("KROD PIC")
    print("------ Loading data ------")
    data_set = 'dataset/COIL20_32.txt'    # K=20 u=1
    # data_set = 'dataset/mnist.txt'    # K=20 u=1
    # data_set = 'dataset/lung.txt'    # K=10 u=0.1
    # data_set = 'dataset/USPS.txt'    # K=20 u=1
    # data_set = 'dataset/Isolet.txt'    # K=25 u=10
    # data_set = 'dataset/TOX.txt'    # K=20 u=10
    # data_set = 'dataset/Jaffe.txt'    # K=10  u=0.1

    # fea, labels = loadData.load_coil100()

    data = np.loadtxt(data_set)
    fea = data[:, :-1]
    labels = data[:, -1]
    print("data_set = %s    data.shape = %s" % (data_set, fea.shape))

    print("------ Normalizing data ------")
    # tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

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
    labels_pred = tool.gacMerging(graphW, initialCluster, groupNumber, 'path', z)
    end = time.time()
    print("time =",end - start)

    nmi = measure.NMI(labels, labels_pred)
    print("nmi =",nmi)
    acc = measure.ACC(labels, labels_pred)
    print("acc =",acc)
    precision_score = measure.precision_score(labels, labels_pred)
    print("precision_score =", precision_score)
