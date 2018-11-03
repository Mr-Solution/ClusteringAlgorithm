# -*- coding:utf-8 -*-

from tool import (measure, loadData)
from sklearn.preprocessing import MinMaxScaler
import DPC
import numpy as np
import time

if __name__ == '__main__':
    print("KROD DPC")
    print("------ Loading data ------")
    # data_set = 'dataset/COIL20_32.txt'    # p=1.5 u=100
    # data_set = 'dataset/mnist.txt'    # p=0.2 u=1
    # data_set = 'dataset/lung.txt'    # p=2 u=0.1
    # data_set = 'dataset/USPS.txt'    # p=0.2 u=1
    # data_set = 'dataset/Isolet.txt'    # p=0.5 u=1
    # data_set = 'dataset/TOX.txt'    # p=2 u=10
    # data_set = 'dataset/Jaffe.txt'    # p=1.5 u=0.1
    fea, labels = loadData.load_coil100()    # p=0.5 u=10

    # data = np.loadtxt(data_set)
    # fea = data[:, :-1]
    # labels = data[:,-1]
    # print("data_set = %s  data.shape = %s" % (data_set, fea.shape))


    print("------ Normalizing data ------")
    # tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    p = 0.5
    u = 10
    groupNumber = len(np.unique(labels))

    print("------ Clustering ------")
    start = time.time()
    labels_pred = DPC.DPC1(fea, groupNumber, p, u)
    end = time.time()
    print("time =", end - start)

    nmi = measure.NMI(labels, labels_pred)
    print("nmi =", nmi)
    acc = measure.ACC(labels, labels_pred)
    print("acc =", acc)
    precision_score = measure.precision_score(labels, labels_pred)
    print("precision_score =", precision_score)
