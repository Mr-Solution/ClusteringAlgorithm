# -*- coding:utf-8 -*-

from tool import (tool, measure, loadData)
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import time

if __name__ == '__main__':
    print("N-Cuts")
    # dataset = 'dataset/COIL20_32.txt'
    dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/USPS.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/TOX.txt'
    # dataset = 'dataset/Jaffe.txt'

    data = np.loadtxt(dataset)
    fea = data[:, :-1]
    labels = data[:, -1]
    print("dataset = %s    data.shape = %s" % (dataset, fea.shape))

    print("------ Normalizing data ------")
    # fea = tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ clustering ------")
    groupNumber = len(np.unique(labels))
    start = time.time()
    clustering = SpectralClustering(n_clusters=groupNumber).fit(fea)
    cl = clustering.labels_
    end = time.time()
    print("time =", end-start)

    print("------ Computing performance measure ------")
    NMI = measure.NMI(labels, cl)
    print("NMI =",  NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
