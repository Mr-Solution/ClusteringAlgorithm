# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from tool import tool,measure
import AGDL
import time

if __name__ == '__main__':
    #dataset = 'dataset/Isolet.txt'
    #dataset = 'dataset/COIL20_32.txt'
    dataset = 'dataset/lung.txt'

    print("GDL    dataset =", dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    print("data.shape =",fea.shape)
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    # dist = tool.rank_dis_c(fea)
    # dist = dist - np.diag(np.diag(dist))

    dist = cdist(fea, fea)

    groupNumber = len(np.unique(labels))
    K = 18    # the number of nearest neighbors for KNN graph
    a = 10
    # cl = AGDL.AGDL(dist, groupNumber, K, v)

    start = time.time()
    cluster = AGDL.AGDL(fea, dist, groupNumber, K, 5, a)
    labels_pred = np.zeros(len(labels), dtype='i')
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            labels_pred[cluster[i][j]] = i
    end = time.time()
    print("time =", end - start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)