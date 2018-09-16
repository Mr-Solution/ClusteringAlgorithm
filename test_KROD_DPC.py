# -*- coding:utf-8 -*-

import ClusteringAlgorithm.tool.tool as tool
import numpy as np

if __name__  == '__main__':
    data = np.loadtxt('fea.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    u = 1
    p = 1.5
    groupNumber = len(np.unique(labels))