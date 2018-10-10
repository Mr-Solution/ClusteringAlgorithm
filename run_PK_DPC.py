# -*- coding:utf-8 -*-

import knnDPC
import numpy as np
from sklearn.decomposition import PCA
from tool import tool,measure
# import tool.PCA

if __name__ == '__main__':
    print("hello PK DPC")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    # fea,b,c = tool.PCA.pca(fea, 150)
    pca = PCA(n_components=150)
    fea = pca.fit_transform(fea)

    K = 5
    groupNumber = len(np.unique(labels))

    cl = knnDPC.knnDPC2(fea, groupNumber, K)

    nmi = measure.NMI(labels, cl)
    print("nmi =",nmi)
    acc = measure.ACC(labels, cl)  # 计算ACC需要label和labels_pred 在同一个区间[1,20]
    print("acc =",acc)

    print("world")