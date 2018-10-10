# -*- coding:utf-8 -*-

from tool import tool,measure
import DPC
import numpy as np
import time

if __name__ == '__main__':
    print("hello KROD DPC")
    #data = np.loadtxt('dataset/COIL20_32.txt')
    #data = np.loadtxt('dataset/mnist.txt')
    # data = np.loadtxt('dataset/mnist.txt')
    # data = np.loadtxt('dataset/USPS.txt')   # p = 0.8 u = 1
    # data = np.loadtxt('dataset/Isolet.txt')   # p = 0.52 u = 1
    dataset = 'dataset/lung.txt'

    print("KROD_DPC    dataset =",dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    u = 0.1
    p = 2
    groupNumber = len(np.unique(labels))

    start = time.time()
    cl = DPC.DPC1(fea, groupNumber, p, u)
    end = time.time()
    print("time =",end - start)

    nmi = measure.NMI(labels, cl)
    print("nmi =",nmi)
    acc = measure.ACC(labels, cl)
    print("acc =",acc)
    print('world')