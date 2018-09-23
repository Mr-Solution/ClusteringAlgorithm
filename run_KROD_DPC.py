# -*- coding:utf-8 -*-

import tool.tool as tool
import DPC
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    u = 1
    p = 1.5
    groupNumber = len(np.unique(labels))

    cl = DPC.DPC(fea, groupNumber, p, u)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')
    # cl 结果同matlab程序一致，但是NMI不一致，python 低了接近 1%

