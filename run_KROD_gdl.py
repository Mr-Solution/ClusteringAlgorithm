# -*- coding:utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tool import (tool, measure, loadData)
import AGDL
import GDL
import time


if __name__ == '__main__':
    print("KROD-GDL")
    # data_set = 'dataset/COIL20_32.txt'    # K=5 v=10
    # data_set = 'dataset/Isolet.txt'    # K=21  v=10
    # data_set = 'dataset/Jaffe.txt'    # K=10 u=1 v=1            k=15 v=10
    # data_set = 'dataset/lung.txt'    # K=10  u=0.1  v=0.1
    # data_set = 'dataset/mnist.txt'    # K=25 u=100 v=10
    # data_set = 'dataset/TOX.txt'    # K=10 u=1 v=10
    # data_set = 'dataset/USPS.txt'    # K=10 u=100 v=100

    # data = np.loadtxt(data_set)
    # fea = data[:, :-1]
    # labels = data[:,-1]
    # print("data_set = %s    data.shape = %s" % (data_set, fea.shape))

    fea, labels = loadData.load_coil100()    # K=25 u=100 v=0.1
    print("data_set = COIL100    data.shape =", fea.shape)

    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    # u = 100
    # dist = tool.rank_dis_c(fea, u)
    dist = tool.rank_order_dis(fea)

    group_number = len(np.unique(labels))
    K = 25    # the number of nearest neighbors for KNN graph
    v = 0.1

    print("------ Clustering ------")
    start = time.time()
    labels_pred = GDL.gdl(dist, group_number, K, v, True)
    end = time.time()
    print("time =", end - start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
    precision_score = measure.precision_score(labels, labels_pred)
    print("precision_score =", precision_score)
