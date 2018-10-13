# -*- coding:utf-8 -*-

from tool import tool,measure
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    print("hello")
    #dataset = 'dataset/COIL20_32.txt'
    #dataset = 'dataset/Isolet.txt'
    #dataset = 'dataset/Jaffe.txt'
    #dataset = 'dataset/lung.txt'
    dataset = 'dataset/mnist.txt'
    #dataset = 'dataset/TOX.txt'
    #dataset = 'dataset/USPS.txt'

    print("kmeans    dataset =", dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    print("data.shape =", data.shape)
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    dist = cdist(fea, fea)

    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    # 用 sklearn 库的 kmeans 算法, 结果比 matlab 的 litekmeans 好一些
    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    kmeans.fit(fea)
    labels_pred = kmeans.labels_

    NMI = measure.NMI(labels, labels_pred)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, labels_pred)
    print("ACC =", ACC)
    print('world')