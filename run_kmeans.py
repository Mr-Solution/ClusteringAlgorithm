# -*- coding:utf-8 -*-

from tool import tool, measure, loadData
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time

if __name__ == '__main__':
    print("hello")
    # dataset = 'dataset/COIL20_32.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/Jaffe.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/TOX.txt'
    # dataset = 'dataset/USPS.txt'

    # print("kmeans    dataset =", dataset)
    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # print("data.shape =", data.shape)
    # labels = data[:, -1]

    fea, labels = loadData.load_coil100()
    fea = tool.data_Normalized(fea)

    dist = cdist(fea, fea)

    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    start = time.time()
    # 用 sklearn 库的 kmeans 算法, 结果比 matlab 的 litekmeans 好一些
    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    kmeans.fit(fea)
    labels_pred = kmeans.labels_
    end = time.time()
    print('time =', end-start)

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
    print('world')
