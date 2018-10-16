# -*- coding:utf-8 -*-

from tool import tool, measure, loadData
#import nmf_LP
import numpy as np
from sklearn import metrics
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

if __name__ == '__main__':
    print("NMF_LP")

    # dataset = 'dataset/COIL20_32.txt'
    # dataset = 'dataset/Isolet.txt'
    # dataset = 'dataset/Jaffe.txt'
    # dataset = 'dataset/lung.txt'
    # dataset = 'dataset/mnist.txt'
    # dataset = 'dataset/TOX.txt'
    # dataset = 'dataset/USPS.txt'

    # data = np.loadtxt(dataset)
    # fea = data[:, :-1]
    # labels = data[:, -1]

    fea, labels = loadData.load_coil100()

    # fea = tool.data_Normalized(fea)
    tool.data_Normalized(fea)

    # p = 2
    groupNumber = len(np.unique(labels))

    print("------ Clustering ------")

    # cl = nmf_LP.predict_nmf_lp(fea, groupNumber)

    # NMF: Non-negative Matrix Factorization
    # NMF 是一种提取特征或降维的方法，这里用 sklearn 包中提供的 api
    model = NMF(n_components=10)
    #model.fit(fea)
    W = model.fit_transform(fea)
    H = model.components_
    #V = model.components_

    kmeans = KMeans(init='k-means++', n_clusters=groupNumber, n_init=20)
    kmeans.fit(W)
    cl = kmeans.labels_

    NMI = measure.NMI(labels, cl)
    print("NMI =", NMI)
    ACC = measure.ACC(labels, cl)
    print("ACC =", ACC)
