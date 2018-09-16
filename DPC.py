# -*- coding:utf-8 -*-

from tool import tool
import numpy as np

def DPC(fea, k, percent, sigma):
    NUC  = k;

    dist = tool.rank_dis_c(fea, sigma)
    dist = dist - np.diag(np.diag(dist))
    ND = dist.shape[0]
    N = pow(ND, 2)

    for i in range(ND):
        dist[i,i] = 0;

    """
    cluster_dp1.m line18
    """

    pass