# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import time

if __name__ == '__main__':
    print("k-means")
    dataset = 'dataset/COIL20_32.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/Jaffe.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/TOX.txt'
    # dataset = 'dataset/USPS.txt'
    # dataset = 'dataset/ORL.txt'
    # dataset = 'dataset/Yale32.txt'
    # dataset = 'dataset/YaleB32.txt'
    # dataset = 'dataset/yeast.txt'

    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:, -1]
    print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    # fea, labels = loadData.load_coil100()

    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ clustering ------")
    groupNumber = len(np.unique(labels))
    dist = cdist(fea, fea)
    start = time.time()
    # 用 sklearn 库的 kmeans 算法, 结果比 matlab 的 litekmeans 好一些
    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=10)
    kmeans.fit(fea)
    labels_pred = kmeans.labels_
    end = time.time()
    print("time =", end-start)

    print("------ Computing performance measure ------")
    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
