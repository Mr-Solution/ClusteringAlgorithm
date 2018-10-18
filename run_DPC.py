# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import DPC
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    print("DPC Algorithm")

    # dataset = 'dataset/COIL20_32.txt'    # p=2
    # dataset = 'dataset/Isolet.txt'    # p=2
    # dataset = 'dataset/Jaffe.txt'    # p=1.5
    # dataset = 'dataset/lung.txt'    # p=2
    # dataset = 'dataset/mnist.txt'    # p=0.5
    # dataset = 'dataset/TOX.txt'    # p=2
    # dataset = 'dataset/USPS.txt'    # p=2
    fea, labels = loadData.load_coil100()

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # print("dataset = %s    data.shape = %s" % (dataset, fea.shape))
    # labels = data[:, -1]
    fea = tool.data_Normalized(fea)

    p = 2
    groupNumber = len(np.unique(labels))

    print("------ clustering ------")
    cl = DPC.DPC2(fea, groupNumber, p)

    NMI = measure.NMI(labels, cl)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
