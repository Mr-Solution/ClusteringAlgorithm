# -*- coding:utf-8 -*-

from tool import tool,measure
import numpy as np
from sklearn.cluster import SpectralClustering
import time

if __name__ == '__main__':

    dataset = 'dataset/COIL20_32.txt'
    #dataset = 'dataset/mnist.txt'
    #dataset = 'dataset/lung.txt'
    #dataset = 'dataset/USPS.txt'
    #dataset = 'dataset/Isolet.txt'
    #dataset = 'dataset/TOX.txt'
    #dataset = 'dataset/Jaffe.txt'

    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)
    print("Ncuts------dataset :",dataset)
    print("data.shape :", fea.shape)

    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    # 用 sklearn 库的 kmeans 算法, 结果比 matlab 的 litekmeans 好一些
    # kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    # kmeans.fit(fea)
    # cl = kmeans.labels_
    start = time.time()
    clustering = SpectralClustering(n_clusters=groupNumber).fit(fea)
    cl = clustering.labels_
    end = time.time()
    print("time =", end-start)

    NMI = measure.NMI(labels, cl)
    print("NMI =",  NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)