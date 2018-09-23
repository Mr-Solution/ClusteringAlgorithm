# -*- coding:utf-8 -*-

import tool.tool as tool
import numpy as np

if __name__ == '__main__':
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)


    # NMI = metrics.adjusted_mutual_info_score(cl, labels)
    # print(NMI)
    print('world')