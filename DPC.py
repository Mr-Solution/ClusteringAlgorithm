# -*- coding:utf-8 -*-

from tool import tool
import numpy as np
import copy
from scipy.spatial import distance
#from scipy.spatial.distance import cdist
# from scipy.sparse import coo_matrix

"""
ROD DPC
fea : data
k : the expected number of groups
percent, sigma DPC算法参数
"""
def DPC1(fea, k, percent, sigma):
    NUC = k;

    dist = tool.rank_dis_c(fea, sigma)
    dist = dist - np.diag(np.diag(dist))
    sample_num = dist.shape[0]
    N = pow(sample_num, 2)

    dist = dist - np.diag(np.diag(dist))

    # dc is a cutoff distance, 通过调参percent，选取前percent%
    position = int(np.round(N*percent/100))
    xx = np.reshape(dist, N, 1)
    sda = np.sort(xx)
    dc = sda[position - 1]
    """
    sparsedist = coo_matrix(np.triu(dist))
    sda = np.sort(sparsedist.data)
    position = int(np.round(len(sda)*percent/100))
    dc = sda[position - 1]
    """
    print("dc =",dc)
    #rho = np.zeros(sample_num)

    """
    for i in range(sample_num-1):
        for j in range(i+1, sample_num):
            # rho[i] = rho[i] + np.exp(-pow(dist[i,j]/dc, 2))
            # rho[j] = rho[j] + np.exp(-pow(dist[i,j]/dc, 2))
            # 涉及浮点数运算，(a/b)*(a/b)和(a/b)^2的结果是不同的，中间步骤保留的小数位数不同。
            rho[i] = rho[i] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
            rho[j] = rho[j] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
    """
    """用下面两行替换"""
    tmp = np.triu(np.exp(-(dist/dc)*(dist/dc)), 1)
    rho = np.sum(tmp, axis=0) + np.sum(tmp, axis=1)

    maxd = np.max(dist)

    ordrho = np.argsort(-rho)
    delta = np.zeros(sample_num)
    delta[ordrho[1]] = -1
    nneigh = np.zeros(sample_num)

    # 这段写的什么鬼，考虑加速一下
    for i in range(1, sample_num):
        delta[ordrho[i]] = maxd
        for j in range(i):
            if dist[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                delta[ordrho[i]] = dist[ordrho[i], ordrho[j]]
                nneigh[ordrho[i]] = ordrho[j]

    delta[ordrho[0]] = np.max(delta)

    NCLUST = 0
    cl = -np.ones(sample_num)
    rank = rho * delta
    index = np.argsort(-rank)
    # icl = index[:NUC]  icl 根本没用到嘛
    for i in range(NUC):
        NCLUST += 1
        cl[index[i]] = NCLUST
        #icl[NCLUST] = index[i] 修改了matlab代码

    # 可以用列表运算加速
    for i in range(sample_num):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    cl = cl - 1
    halo = copy.deepcopy(cl)

    if NCLUST > 1:
        bord_rho = np.zeros(NCLUST)
        """
        for i in range(sample_num-1):
            for j in range(i+1, sample_num):
                if cl[i] != cl[j] and dist[i,j] <= dc:
                    rho_aver = (rho[i] + rho[j])/2
                    if rho_aver > bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver > bord_rho[int(cl[j])]:
                        bord_rho[int(cl[j])] = rho_aver
        """
        xindices,yindices = np.where(dist <= dc)
        for x,y in zip(xindices,yindices) :
            # 只考虑上三角
            if y>x and cl[x] != cl[y]:
                rho_aver = (rho[x] + rho[y])/2
                if rho_aver > bord_rho[int(cl[x])]:
                    bord_rho[int(cl[x])] = rho_aver
                if rho_aver > bord_rho[int(cl[y])]:
                    bord_rho[int(cl[y])] = rho_aver


        # 不太好办，但是应该可以加速
        for i in range(sample_num):
            if rho[i] < bord_rho[int(cl[i])]:
                halo[i] = 0

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(sample_num):
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

    #rho = np.zeros(ND)
    """
    for i in range(ND-1):
        for j in range(i+1, ND):
            # rho[i] = rho[i] + np.exp(-pow(dist[i,j]/dc, 2))
            # rho[j] = rho[j] + np.exp(-pow(dist[i,j]/dc, 2))
            # 涉及浮点数运算，(a/b)*(a/b)和(a/b)^2的结果是不同的，中间步骤保留的小数位数不同。
            rho[i] = rho[i] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
            rho[j] = rho[j] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
    """
    tmp = np.triu(np.exp(-(dist/dc)*(dist/dc)), 1)
    rho = np.sum(tmp, axis=0) + np.sum(tmp, axis=1)

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
        """
        for i in range(ND-1):
            for j in range(i+1, ND):
                if cl[i] != cl[j] and dist[i,j] <= dc:
                    rho_aver = (rho[i] + rho[j])/2
                    if rho_aver > bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver > bord_rho[int(cl[j])]:
                        bord_rho[int(cl[j])] = rho_aver
        """
        xindices,yindices = np.where(dist <= dc)
        for i in range(len(xindices)):
            x = xindices[i]
            y = yindices[i]
            # 只考虑上三角
            if y>x and cl[x] != cl[y]:
                rho_aver = (rho[x] + rho[y])/2
                if rho_aver > bord_rho[int(cl[x])]:
                    bord_rho[int(cl[x])] = rho_aver
                if rho_aver > bord_rho[int(cl[y])]:
                    bord_rho[int(cl[y])] = rho_aver

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

def knnDPC1(fea, k, K, sigma2):
    NUC = k;

    dist = tool.rank_dis_c(fea, sigma2)
    dist = dist - np.diag(np.diag(dist))
    sample_num = dist.shape[0]
    N = pow(sample_num, 2)

    dist = dist - np.diag(np.diag(dist))

    percent = 2
    # dc is a cutoff distance, 通过调参percent，选取前percent%
    position = int(np.round(N*percent/100))
    xx = np.reshape(dist, N, 1)
    sda = np.sort(xx)
    dc = sda[position - 1]

    rho = np.zeros(sample_num)

    dist_copy = copy.deepcopy(dist)
    dist_copy = np.sort(dist_copy, axis=0)

    for i in range(1,K+1):
        pass

    for i in range(sample_num):
        dist_copy[:,i] = np.sort(dist_copy[:,i])

    for i in range(1,K):
        rho = rho + dist_copy[i,:] * dist_copy[i,:]

    rho = np.exp(-rho/k)

    maxd = np.max(dist)

    ordrho = np.argsort(-rho)
    delta = np.zeros(sample_num)
    #delta = np.ones(sample_num) * maxd
    delta[ordrho[1]] = -1
    nneigh = np.zeros(sample_num)

    # 这段写的什么鬼，考虑加速一下
    for i in range(1, sample_num):
        delta[ordrho[i]] = maxd
        for j in range(i):
            if dist[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                delta[ordrho[i]] = dist[ordrho[i], ordrho[j]]
                nneigh[ordrho[i]] = ordrho[j]


    delta[ordrho[0]] = np.max(delta)

    NCLUST = 0
    cl = -np.ones(sample_num)
    rank = rho * delta
    index = np.argsort(-rank)
    for i in range(NUC):
        NCLUST += 1
        cl[index[i]] = NCLUST

    # 可以用列表运算加速
    for i in range(sample_num):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    cl = cl - 1
    halo = copy.deepcopy(cl)

    if NCLUST > 1:
        bord_rho = np.zeros(NCLUST)
        # for i in range(sample_num-1):
        #     for j in range(i+1, sample_num):
        #         if cl[i] != cl[j] and dist[i,j] <= dc:
        #             rho_aver = (rho[i] + rho[j])/2
        #             if rho_aver > bord_rho[int(cl[i])]:
        #                 bord_rho[int(cl[i])] = rho_aver
        #             if rho_aver > bord_rho[int(cl[j])]:
        #                 bord_rho[int(cl[j])] = rho_aver

        xindices,yindices = np.where(dist <= dc)
        for x,y in zip(xindices,yindices) :
            # 只考虑上三角
            if y>x and cl[x] != cl[y]:
                rho_aver = (rho[x] + rho[y])/2
                if rho_aver > bord_rho[int(cl[x])]:
                    bord_rho[int(cl[x])] = rho_aver
                if rho_aver > bord_rho[int(cl[y])]:
                    bord_rho[int(cl[y])] = rho_aver


        # 不太好办，但是应该可以加速
        for i in range(sample_num):
            if rho[i] < bord_rho[int(cl[i])]:
                halo[i] = 0

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(sample_num):
            if cl[j] == i:
                nc += 1
            if halo[j] == i:
                nh += 1
    print("nc = %d nh = %d"%(nc,nh))
    return cl


def knnDPC2(fea, k, K):
    NUC  = k;

    #dist = tool.rank_dis_c(fea, sigma2)
    dist = distance.cdist(fea, fea)
    dist = dist - np.diag(np.diag(dist))
    ND = dist.shape[0]
    N = pow(ND, 2)

    dist = dist - np.diag(np.diag(dist))

    percent = 2
    # dc is a cutoff distance, 通过调参percent，选取前percent%
    position = int(np.round(N*percent/100))
    xx = np.reshape(dist, N, 1)
    sda = np.sort(xx)
    dc = sda[position - 1]

    rho = np.zeros(ND)

    # for i in range(ND-1):
    #     for j in range(i+1, ND):
    #         # rho[i] = rho[i] + np.exp(-pow(dist[i,j]/dc, 2))
    #         # rho[j] = rho[j] + np.exp(-pow(dist[i,j]/dc, 2))
    #         # 涉及浮点数运算，(a/b)*(a/b)和(a/b)^2的结果是不同的，中间步骤保留的小数位数不同。
    #         rho[i] = rho[i] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
    #         rho[j] = rho[j] + np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))

    dist_copy = copy.deepcopy(dist)
    dist_copy = np.sort(dist_copy, axis=0)

    for i in range(1,K+1):
        pass

    for i in range(ND):
        dist_copy[:,i] = np.sort(dist_copy[:,i])

    for i in range(1,K):
        rho = rho + dist_copy[i,:] * dist_copy[i,:]

    rho = np.exp(-rho/k)

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