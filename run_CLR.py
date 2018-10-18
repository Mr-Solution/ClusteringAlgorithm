# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import CLR
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    print("CLR")

    # dataset = 'dataset/COIL20_32.txt'    # K=5
    # dataset = 'dataset/Isolet.txt'    # K=21
    # dataset = 'dataset/Jaffe.txt'    # K=20
    # dataset = 'dataset/lung.txt'    # K=20
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/TOX.txt'    # K=10
    # dataset = 'dataset/USPS.txt'    # K=20


    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:,-1]
    fea, labels = loadData.load_coil100()
    # print("dataset : %s, shape : %s" % (dataset, fea.shape))
    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    K = 5
    groupNumber = len(np.unique(labels))

    print("------ Clustering ------")
    A0 = tool.constructW_PKN(fea, K)
    # cl, S, evs, cs = CLR.CLR(A0, groupNumber, 0, 1)
    cl, evs, cs = CLR.CLR(A0, groupNumber, 0, 1)

    NMI = measure.NMI(labels, cl)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
