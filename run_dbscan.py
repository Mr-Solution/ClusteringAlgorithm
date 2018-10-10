# -*- coding:utf-8 -*-

import tool.tool as tool
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    print("hello")
    dataset = 'dataset/COIL20_32.txt'

    print("dbscan    dataset =",dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    # 调参 eps min_samples
    clustering = DBSCAN(eps=3, min_samples=3)
    clustering.fit(fea)
    cl = clustering.labels_

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')