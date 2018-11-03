# -*- coding:utf-8 -*-

from tool import tool,measure
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import time

if __name__ == '__main__':
    print("hello")
    # dataset = 'dataset/COIL20_32.txt'  # eps=0.005, min_samples=3
    # dataset = 'dataset/mnist.txt'
    dataset = 'dataset/lung.txt'

    print("dbscan    dataset =",dataset)
    data = np.loadtxt(dataset)
    print("data.shape =",data.shape)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    #dist = cdist(fea, fea)
    a = 1
    dist = tool.rank_dis_c(fea, a)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    start = time.time()
    # 调参 eps min_samples
    # clustering = DBSCAN(eps=3, min_samples=3)
    # clustering.fit(fea)

    #nmi = 0
    #eps = 0
    # for i in range(1,100,1):
    #     e = i* 1e-3
    #     DBSCAN_CLUSTER = DBSCAN(i, min_samples=3, metric='precomputed')
    #     DBSCAN_CLUSTER.fit(dist)
    #     labels_pred = DBSCAN_CLUSTER.labels_
    #     NMI = measure.NMI(labels,labels_pred)
    #     if NMI > nmi:
    #         nmi = NMI
    #         eps = i

    #print("eps =", eps)


    DBSCAN_CLUSTER = DBSCAN(eps=13, min_samples=3, metric='precomputed')
    DBSCAN_CLUSTER.fit(dist)
    labels_pred = DBSCAN_CLUSTER.labels_
    end = time.time()
    print("time =",end - start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
