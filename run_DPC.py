# -*- coding:utf-8 -*-

import tool.tool as tool
import DPC
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    cl = DPC.DPC2(fea, groupNumber, p)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print('world')