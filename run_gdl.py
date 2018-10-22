# -*- coding:utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from tool import (measure, loadData)
# import AGDL
import GDL
import time

if __name__ == '__main__':
    print("GDL")
    print("------ Loading data ------")
    dataset = 'dataset/Isolet.txt'    # K=25 v=10
    # dataset = 'dataset/lung.txt'    # K=15 v=0.1
    # dataset = 'dataset/TOX.txt'    # K=10 v=10
    # dataset = 'dataset/Jaffe.txt'    # K=15 v=10
    # dataset = 'dataset/USPS.txt'    # K=20 v=1
    # dataset = 'dataset/mnist.txt'    # K=25  v=1
    # dataset = 'dataset/COIL20_32.txt'    # K=5  v=1

    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:,-1]
    print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    # fea, labels = loadData.load_coil100()

    print("------ Normalizing data ------")
    # tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    # dist = tool.rank_dis_c(fea)
    # dist = dist - np.diag(np.diag(dist))
    print("------ Clustering ------")
    start = time.time()
    dist = cdist(fea, fea)
    groupNumber = len(np.unique(labels))
    K = 25    # the number of nearest neighbors for KNN graph
    v = 10
    # a = 10
    # cl = AGDL.AGDL(dist, groupNumber, K, v)

    # cluster = AGDL.AGDL(fea, dist, groupNumber, K, 5, a)
    # labels_pred = np.zeros(len(labels), dtype='i')
    # for i in range(len(cluster)):
    #     for j in range(len(cluster[i])):
    #         labels_pred[cluster[i][j]] = i
    labels_pred = GDL.gdl(dist, groupNumber, K, v, True)
    end = time.time()
    print("time =", end - start)

    print("------ Computing performance measure ------")
    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
