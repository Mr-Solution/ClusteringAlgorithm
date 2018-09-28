# -*- coding:utf-8 -*-

import knnDPC
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
import tool.tool
import tool.PCA

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
    print("hello")
    data = np.loadtxt('dataset/COIL20_32.txt')
    fea = data[:, :-1]
    labels = data[:,-1]
    fea = tool.tool.data_Normalized(fea)

    #fea,b,c = tool.PCA.pca(fea, 150)
    pca = PCA(n_components=150)
    fea = pca.fit_transform(fea)

    u = 1
    K = 20
    groupNumber = len(np.unique(labels))

    cl = knnDPC.knnDPC1(fea, groupNumber, K, u)

    NMI = metrics.adjusted_mutual_info_score(cl, labels)
    print(NMI)
    print("world")
