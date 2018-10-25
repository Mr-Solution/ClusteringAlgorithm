# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import time

if __name__ == '__main__':
    print("N-Cuts")
    print("------ Loading data ------")
    # dataset = 'dataset/Isolet.txt'    # n_neighbors=28
    # dataset = 'dataset/lung.txt'    # n_neighbors=15
    # dataset = 'dataset/TOX.txt'    # n_neighbors=5
    # dataset = 'dataset/Jaffe.txt'    # n_neighbors=15
    # dataset = 'dataset/USPS.txt'    # n_neighbors=10
    # dataset = 'dataset/mnist.txt'    # n_neighbors=25
    # dataset = 'dataset/COIL20_32.txt'

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:, -1]
    # print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    fea, labels = loadData.load_coil100()

    print("------ Normalizing data ------")
    # tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ Clustering ------")
    # 因使用 K-Means 聚类，NMI ACC 并不稳定
    groupNumber = len(np.unique(labels))
    start = time.time()
    # 默认的构建相似矩阵方式是高斯核，但是效果很差，这里采用 k 邻近法，调参n_neighbors
    clustering = SpectralClustering(n_clusters=groupNumber, affinity='nearest_neighbors', n_neighbors=100, n_init=groupNumber).fit(fea)
    # clustering = SpectralClustering(n_clusters=groupNumber, n_init=groupNumber).fit(fea)
    cl = clustering.labels_
    end = time.time()
    print("time =", end-start)

    print("------ Computing performance measure ------")
    NMI = measure.NMI(labels, cl)
    print("NMI =",  NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
