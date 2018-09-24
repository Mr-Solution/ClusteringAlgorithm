# -*- coding:utf-8 -*-

from tool import tool
import numpy as np
import copy
from scipy.spatial import distance

"""
ROD DPC
"""
def DPC1(fea, k, percent, sigma):
    NUC  = k;

    dist = tool.rank_dis_c(fea, sigma)
    dist = dist - np.diag(np.diag(dist))
    ND = dist.shape[0]
    N = pow(ND, 2)

    dist = dist - np.diag(np.diag(dist))

    # dc is a cutoff distance, 通过调参percent，选取前percent%
    position = int(np.round(N*percent/100))
    xx = np.reshape(dist, N, 1)
    sda = np.sort(xx)
    dc = sda[position - 1]

    rho = np.zeros(ND)

    for i in range(ND-1):
        for j in range(i+1, ND):
            # rho[i] = rho[i] + np.exp(-pow(dist[i,j]/dc, 2))
            # rho[j] = rho[j] + np.exp(-pow(dist[i,j]/dc, 2))
            # 涉及浮点数运算，(a/b)*(a/b)和(a/b)^2的结果是不同的，中间步骤保留的小数位数不同。
            rho[i] = rho[i] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
            rho[j] = rho[j] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))

    maxd = np.max(dist)

    ordrho = np.argsort(-rho)
    delta = np.zeros(ND)
    delta[ordrho[1]] = -1
    nneigh = np.zeros(ND)

    # 这段写的什么鬼，考虑加速一下
    for i in range(1, ND):
        delta[ordrho[i]] = maxd
        for j in range(i):
            if dist[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                delta[ordrho[i]] = dist[ordrho[i], ordrho[j]]
                nneigh[ordrho[i]] = ordrho[j]

    delta[ordrho[0]] = np.max(delta)

    NCLUST = 0
    cl = -np.ones(ND)
    rank = rho * delta
    index = np.argsort(-rank)
    # icl = index[:NUC]  icl 根本没用到嘛
    for i in range(NUC):
        NCLUST += 1
        cl[index[i]] = NCLUST
        #icl[NCLUST] = index[i] 修改了matlab代码

    # 可以用列表运算加速
    for i in range(ND):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    cl = cl - 1
    halo = copy.deepcopy(cl)

    if NCLUST > 1:
        bord_rho = np.zeros(NCLUST)
        for i in range(ND-1):
            for j in range(i+1, ND):
                if cl[i] != cl[j] and dist[i,j] <= dc:
                    rho_aver = (rho[i] + rho[j])/2
                    if rho_aver > bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver > bord_rho[int(cl[j])]:
                        bord_rho[int(cl[j])] = rho_aver

        # 不太好办，但是应该可以加速
        for i in range(ND):
            if rho[i] < bord_rho[int(cl[i])]:
                halo[i] = 0

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(ND):
            if cl[j] == i:
                nc += 1
            if halo[j] == i:
                nh += 1
    print("nc = %d nh = %d"%(nc,nh))
    return cl

"""
origin DPC
"""
def DPC2(fea, k, percent):
    NUC  = k;
    dist = distance.cdist(fea, fea)
    dist = dist - np.diag(np.diag(dist))
    ND = dist.shape[0]
    N = pow(ND, 2)
    dist = dist - np.diag(np.diag(dist))

    # dc is a cutoff distance, 通过调参percent，选取前percent%
    position = int(np.round(N*percent/100))
    xx = np.reshape(dist, N, 1)
    sda = np.sort(xx)
    dc = sda[position - 1]

    rho = np.zeros(ND)

    for i in range(ND-1):
        for j in range(i+1, ND):
            # rho[i] = rho[i] + np.exp(-pow(dist[i,j]/dc, 2))
            # rho[j] = rho[j] + np.exp(-pow(dist[i,j]/dc, 2))
            # 涉及浮点数运算，(a/b)*(a/b)和(a/b)^2的结果是不同的，中间步骤保留的小数位数不同。
            rho[i] = rho[i] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
            rho[j] = rho[j] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))

    maxd = np.max(dist)

    ordrho = np.argsort(-rho)
    delta = np.zeros(ND)
    delta[ordrho[1]] = -1
    nneigh = np.zeros(ND)

    # 这段写的什么鬼，考虑加速一下
    for i in range(1, ND):
        delta[ordrho[i]] = maxd
        for j in range(i):
            if dist[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                delta[ordrho[i]] = dist[ordrho[i], ordrho[j]]
                nneigh[ordrho[i]] = ordrho[j]

    delta[ordrho[0]] = np.max(delta)

    NCLUST = 0
    cl = -np.ones(ND)
    rank = rho * delta
    index = np.argsort(-rank)
    # icl = index[:NUC]  icl 根本没用到嘛
    for i in range(NUC):
        NCLUST += 1
        cl[index[i]] = NCLUST
        #icl[NCLUST] = index[i] 修改了matlab代码

    # 可以用列表运算加速
    for i in range(ND):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    cl = cl - 1
    halo = copy.deepcopy(cl)

    if NCLUST > 1:
        bord_rho = np.zeros(NCLUST)
        for i in range(ND-1):
            for j in range(i+1, ND):
                if cl[i] != cl[j] and dist[i,j] <= dc:
                    rho_aver = (rho[i] + rho[j])/2
                    if rho_aver > bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver > bord_rho[int(cl[j])]:
                        bord_rho[int(cl[j])] = rho_aver

        # 不太好办，但是应该可以加速
        for i in range(ND):
            if rho[i] < bord_rho[int(cl[i])]:
                halo[i] = 0

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(ND):
            if cl[j] == i:
                nc += 1
            if halo[j] == i:
                nh += 1
    print("nc = %d nh = %d"%(nc,nh))
    return cl