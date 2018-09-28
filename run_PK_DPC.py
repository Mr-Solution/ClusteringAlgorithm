# -*- coding:utf-8 -*-

import knnDPC
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
import tool.tool
# import tool.PCA

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.tool.data_Normalized(fea)

    # fea,b,c = tool.PCA.pca(fea, 150)
    pca = PCA(n_components=150)
    fea = pca.fit_transform(fea)

    K = 5
    groupNumber = len(np.unique(labels))

    cl = knnDPC.knnDPC2(fea, groupNumber, K)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print("world")