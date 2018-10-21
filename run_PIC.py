# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    print("PIC")
    """ load data """
    # dataset = 'dataset/COIL20_32.txt'    # K=5  v=1
    # dataset = 'dataset/Isolet.txt'    # K=25  v=10
    # dataset = 'dataset/Jaffe.txt'    # K=15  v=10
    # dataset = 'dataset/lung.txt'    # K=15  v=10
    # dataset = 'dataset/mnist.txt'    # K=25  v=1
    # dataset = 'dataset/TOX.txt'    # K=15  v=10
    # dataset = 'dataset/USPS.txt'    # K=20  v=1

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:, -1]
    # print("dataset = %s    data.shape = %s" % (dataset, fea.shape))
    fea, labels = loadData.load_coil100()     # K=10  v=1

    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ Clustering ------")
    start = time.time()
    # u = 1
    dist = cdist(fea, fea)
    dist = dist - np.diag(np.diag(dist))

    K = 10
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
    end = time.time()
    print("time =", end - start)

    print("------ Computing performance measure ------")
    NMI = measure.NMI(labels, cl)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
