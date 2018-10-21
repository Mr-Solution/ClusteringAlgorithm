# -*- coding:utf-8 -*-

import knnDPC
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tool import (loadData, measure)
# import tool.PCA

if __name__ == '__main__':
    print("PK DPC")
    # dataset = 'dataset/COIL20_32.txt'    # K=10
    # dataset = 'dataset/Isolet.txt'    # K=5
    # dataset = 'dataset/Jaffe.txt'    # K=5
    # dataset = 'dataset/lung.txt'    # K=25
    # dataset = 'dataset/mnist.txt'    # K=15
    # dataset = 'dataset/TOX.txt'    # K=10
    # dataset = 'dataset/USPS.txt'    # K=25

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:, -1]
    # print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    fea, labels = loadData.load_coil100()
    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ PCA decomposition ------")
    # fea,b,c = tool.PCA.pca(fea, 150)
    pca = PCA(n_components=150)
    fea = pca.fit_transform(fea)
    print("fea.shape =",fea.shape)

    K = 5
    groupNumber = len(np.unique(labels))

    print("------ Clustering ------")
    start = time.time()
    cl = knnDPC.knnDPC2(fea, groupNumber, K)
    end = time.time()
    print("time =", end - start)

    print("------ Computing performance measure ------")
    nmi = measure.NMI(labels, cl)
    print("nmi =", nmi)
    acc = measure.ACC(labels, cl)  # 计算ACC需要label和labels_pred 在同一个区间[1,20]
    print("acc =", acc)
