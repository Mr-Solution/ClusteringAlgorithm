# -*- coding:utf-8 -*-

import DPC
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tool import tool, measure, loadData
import time
#import tool.PCA

"""
def pca(data_mat, top_n_feature = 999) :
    avg_value = np.mean(data_mat, axis=0)
    cent_mat = data_mat - avg_value
    cov_mat = np.cov(cent_mat, rowvar = 0)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    #print(eig_val)
    eig_val_index = np.argsort(eig_val)  # 从小到大排序
    # -1 切片函数step是负数，意思是倒着切，取大的
    eig_val_index = eig_val_index[-1 : -(top_n_feature + 1) : -1]
    # 保存投影矩阵，中心化的矩阵与投影矩阵相乘，即可得到降维矩阵
    projection_matrix = eig_vec[:, eig_val_index]
    #print("eig_val: %s, top_n_feature: %s, eig_val_index： %s,   projection_matrix: %s"\
    #      %(eig_val, top_n_feature, eig_val_index, projection_matrix))

    low_data_mat= np.dot(cent_mat, projection_matrix)

    #还原原来矩阵（只能近似还原）
    rM = np.dot(low_data_mat, projection_matrix.T) + avg_value
    print("--------------------------")
    print("rM  ===== ")
    print(rM)
    print("--------------------------")
    #
    return low_data_mat, avg_value, projection_matrix
"""

if __name__ == '__main__':
    print("KROD_PCA_KNN_DPC")
    print("------ Loading data ------")
    # data_set = 'dataset/COIL20_32.txt'    # K=20 u=1
    data_set = 'dataset/mnist.txt'    # K=20 u=1
    # data_set = 'dataset/lung.txt'    # K=10 u=0.1
    # data_set = 'dataset/USPS.txt'    # K=20 u=1
    # data_set = 'dataset/Isolet.txt'    # K=25 u=10
    # data_set = 'dataset/TOX.txt'    # K=20 u=10
    # data_set = 'dataset/Jaffe.txt'    # K=10  u=0.1
    # fea, labels = loadData.load_coil100()

    data = np.loadtxt(data_set)
    fea = data[:, :-1]
    labels = data[:,-1]
    print("data_set = %s    data.shape = %s" % (data_set, fea.shape))

    print("------ Normalizing data ------")
    # tool.data_Normalized(fea)
    Normalizer = MinMaxScaler()
    Normalizer.fit(fea)
    fea = Normalizer.transform(fea)

    print("------ Decomposition ------")
    #fea,b,c = PCA.pca(fea, 150)
    pca = PCA(n_components=150)
    fea = pca.fit_transform(fea)
    print("fea.shape =", fea.shape)

    K = 20
    u = 1
    group_number = len(np.unique(labels))

    print("------ Clustering ------")
    start = time.time()
    labels_pred = DPC.knnDPC1(fea, group_number, K, u)
    end = time.time()
    print("time =", end - start)

    nmi = measure.NMI(labels, labels_pred)
    print("nmi =", nmi)
    acc = measure.ACC(labels, labels_pred)
    print("acc =", acc)
    precision_score = measure.precision_score(labels, labels_pred)
    print("precision_score =", precision_score)
