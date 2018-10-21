# -*- coding:utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tool import (tool, measure, loadData)
import AGDL
import GDL
import time


if __name__ == '__main__':
    print("KROD-GDL")

    # dataset = 'dataset/COIL20_32.txt'    # K=5
    # dataset = 'dataset/Isolet.txt'    # K=21
    # dataset = 'dataset/Jaffe.txt'    # K=20
    # dataset = 'dataset/lung.txt'    # K=20
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/TOX.txt'    # K=10
    # dataset = 'dataset/USPS.txt'    # K=20

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:,-1]
    # print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    fea, labels = loadData.load_coil100()

    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    u = 0.1
    dist = tool.rank_dis_c(fea, u)
    dist = dist - np.diag(np.diag(dist))

    groupNumber = len(np.unique(labels))
    K = 20    # the number of nearest neighbors for KNN graph
    v = 1
    a = 10

    print("------ Clustering ------")
    start = time.time()
    labels_pred = GDL.gdl(dist, groupNumber, K, v, True)
    end = time.time()
    print("time =", end - start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
