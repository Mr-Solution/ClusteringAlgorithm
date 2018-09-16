# -*- coding:utf-8 -*-

"""
some functions
"""
import numpy as np

def data_Normalized(data):
    """
    :param data:
    :return: normalized data
    row: samples
    cul: features
    """
    m, n = data.shape
    b1 = []
    b2 = []
    # 遍历每一个特征
    for i in range(n):
        amax = np.max(data[:,i])
        amin = np.min(data[:,i])
        for j in range(m):
            data[j, i] = (data[j,i] - amin)/(amax - amin)

        b1.append(amin)
        b2.append(amax - amin)

    data = np.where(np.isnan(data), 0, data)
    return data, b1, b2
