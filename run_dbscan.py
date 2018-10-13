# -*- coding:utf-8 -*-

from tool import tool,measure
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import time

if __name__ == '__main__':
    print("hello")
    #dataset = 'dataset/COIL20_32.txt'  # e=4.5  m=3  结果噪音点有些多
    #dataset = 'dataset/Isolet.txt'
    dataset = 'dataset/lung.txt'

    print("dbscan    dataset =",dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    dist = cdist(fea, fea)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    start = time.time()
    # 调参 eps min_samples
    DBSCAN_CLUSTER = DBSCAN(eps=11, min_samples=4, metric='precomputed')
    DBSCAN_CLUSTER.fit(dist)
    labels_pred = DBSCAN_CLUSTER.labels_
    end = time.time()
    print("time =",end-start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =",NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)

    dd = DBSCAN(eps=10, min_samples=4)
    dd.fit(fea)
    labels_pred = dd.labels_
    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)

    print('world')