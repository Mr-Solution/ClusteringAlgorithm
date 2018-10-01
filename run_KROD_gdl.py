# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics
import tool.tool as tool
import GDL


if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    dist = tool.rank_dis_c(fea)
    dist = dist - np.diag(np.diag(dist))
    groupNumber = len(np.unique(labels))
    K=5
    v = 1
    cl = GDL.gdl(dist, groupNumber, K, v)
    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')