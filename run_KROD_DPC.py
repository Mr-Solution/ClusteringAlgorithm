# -*- coding:utf-8 -*-

import tool.tool as tool
import DPC
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    fea = np.asarray(fea)
    labels = data[:,-1]
    labels = np.asarray(labels)
    fea = tool.data_Normalized(fea)
    fea = np.asarray(fea)

    u = 1
    p = 1.5
    groupNumber = len(np.unique(labels))

    cl = DPC.DPC(fea, groupNumber, p, u)

    print('world')
    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    # print(NMI)
    # cl 结果同matlab程序一致，但是NMI不一致，python 低了接近 1%

