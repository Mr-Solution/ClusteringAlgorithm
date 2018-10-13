# -*- coding:utf-8 -*-

from tool import tool,measure
import DPC
import numpy as np
import time

if __name__ == '__main__':
    print("hello KROD DPC")
    #dataset = 'dataset/COIL20_32.txt'
    #dataset = 'dataset/mnist.txt'
    #dataset = 'dataset/lung.txt'
    #dataset = 'dataset/USPS.txt'
    #dataset = 'dataset/Isolet.txt'
    dataset = 'dataset/TOX.txt'
    #dataset = 'dataset/Jaffe.txt'

    print("KROD_DPC    dataset =",dataset)
    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.data_Normalized(fea)

    u = 10
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