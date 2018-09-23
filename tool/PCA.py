# -*- coding:utf-8 -*-

import numpy as np

def pca(dataSet,  dim = 10) :
    mean_val = np.mean(dataSet, axis=0)
    cent_mat = dataSet - mean_val
    cov_mat = np.cov(cent_mat, rowvar = 0)
    eig_val, eig_vec = np.linalg.eig(cov_mat)

    eig_val_index = np.argsort(eig_val)  # 从小到大排序
    # -1 切片函数step是负数，意思是倒着切，取大的
    eig_val_index = eig_val_index[ : -(dim + 1) : -1]
    # 保存投影矩阵，中心化的矩阵与投影矩阵相乘，即可得到降维矩阵
    projection_matrix = eig_vec[:, eig_val_index]

    low_data_mat = np.dot(cent_mat, projection_matrix)

    #还原原来矩阵（只能近似还原）
    # rM = np.dot(low_data_mat, projection_matrix.T) + mean_val
    # print("--------------------------")
    # print("rM  ===== ")
    # print(rM)
    # print("--------------------------")

    return low_data_mat, mean_val, projection_matrix