# -*- coding:utf-8 -*-

import tool.tool as tool
import DPC
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    # 用 sklearn 库的 kmeans 算法, 结果比 matlab 的 litekmeans 好一些
    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    kmeans.fit(fea)
    cl = kmeans.labels_

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')