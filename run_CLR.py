# -*- coding:utf-8 -*-

import tool.tool as tool
import CLR
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    K = 5
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    A0 = tool.constructW_PKN(fea, K)
    cl, S, evs, cs = CLR.CLR(A0, groupNumber, 0, 1)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')