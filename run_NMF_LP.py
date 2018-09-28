# -*- coding:utf-8 -*-

import tool.tool as tool
#import nmf_LP
import numpy as np
from sklearn import metrics
from sklearn.decomposition import NMF
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

    # cl = nmf_LP.predict_nmf_lp(fea, groupNumber)

    # NMF: Non-negative Matrix Factorization
    # NMF 是一种提取特征或降维的方法，这里用 sklearn 包中提供的 api
    model = NMF(n_components=10)
    #model.fit(fea)
    W = model.fit_transform(fea)
    H = model.components_
    #V = model.components_

    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    kmeans.fit(W)
    cl = kmeans.labels_

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')