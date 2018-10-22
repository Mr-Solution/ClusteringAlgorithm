# -*- coding:utf-8 -*-

from tool import (measure, loadData)
from sklearn.preprocessing import MinMaxScaler
import DPC
import numpy as np
import time

if __name__ == '__main__':
    print("hello KROD DPC")
    print("------ Loading data ------")
    # dataset = 'dataset/COIL20_32.txt'
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/USPS.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/TOX.txt'
    # dataset = 'dataset/Jaffe.txt'
    fea, labels = loadData.load_coil100()

    # print("KROD_DPC    dataset =",dataset)
    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:,-1]
    # tool.data_Normalized(fea)
    print("------ Normalizing data ------")
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    p = 0.5
    u = 10
    groupNumber = len(np.unique(labels))

    print("------ Clustering ------")
    start = time.time()
    cl = DPC.DPC1(fea, groupNumber, p, u)
    end = time.time()
    print("time =", end - start)

    nmi = measure.NMI(labels, cl)
    print("nmi =", nmi)
    acc = measure.ACC(labels, cl)
    print("acc =", acc)
