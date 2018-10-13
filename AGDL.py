import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.misc import *
#from tool import tool
from scipy.sparse import coo_matrix

def gacBuildDigraph(distance_matrix, K, a):
    # NN indices
    N = distance_matrix.shape[0]
    sortedDist = np.sort(distance_matrix)
    NNIndex = np.argsort(distance_matrix)
    NNIndex = NNIndex[:, :K + 1]
    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K + 1, 4)], 0))*a
    tmpNNdist = np.min(sortedDist[:, 1:], axis=1)

    while np.any(np.exp(-tmpNNdist / sig2) < 1e-5):
        sig2 = 2 * sig2

    ND = sortedDist[:, 1:K+1]
    NI = NNIndex[:, 1:K+1]
    XI = np.tile(np.arange(N).reshape(N, -1), (1, K))
    graphW = coo_matrix(
        (np.exp(-ND.reshape(-1, order='F') * (1 / sig2)), (XI.reshape(-1, order='F'), NI.reshape(-1, order='F'))),
        shape=(N, N)).todense()
    graphW = graphW - np.diag(np.diag(graphW))

    return graphW, NNIndex

""" 
生成权重矩阵 
欧氏距离的权重矩阵应该是对称的
ROD的权重矩阵应该是不对称的
"""
def w_matrix(data, distance, indices, Ks, a=1):

    n = len(data)
    weight_matrix = np.zeros([n, n])
    sigma2 = (a / n / Ks) * np.linalg.norm(distance)**2


    # sig2 = np.mean(np.mean(distance[:, :Ks], 0))
    # tmpNNdist = np.min(distance, axis=1)
    # while np.any(np.exp(-tmpNNdist / sig2) < 1e-5):
    #     sig2 = 10 * sig2
    # sigma2 = sig2

    if Ks==1:
        for i in range(n):
            #for j in range(n):
            j=indices[i][0]
            weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)
    else:
        for i in range(n):
            #for j in range(n):
            for j in indices[i]:
                weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)

    return weight_matrix, sigma2

""" 最近邻图 """
def k0graph(distance, a=1):

    #W, sigma2 = w_matrix(data, distance, indices, 1, a)
    W, sigma2 = gacBuildDigraph(distance, 1, a)
    Vc = []
    x,y = np.where(W>0)

    for i in range(len(x)):
        x_index, y_index = -1,-1
        for k in range(len(Vc)):
            if y[i] in Vc[k]:
                y_index = k
            if x[i] in Vc[k]:
                x_index = k

        if x_index < 0 and y_index < 0:
            Vc.append([x[i],y[i]])
        elif x_index >= 0 and y_index < 0:
            Vc[x_index].append(y[i])
        elif x_index < 0 and y_index >=0:
            Vc[y_index].append(x[i])
        elif x_index == y_index:
            continue
        else:
            Vc[x_index].extend(Vc[y_index])
            del Vc[y_index]

    return Vc

"""
Vc : 初始化的簇，二维list
W ： 权重矩阵
"""
def getAffinityMaxtrix(Vc,W):

    nc = len(Vc)
    # nc 是初始化的簇的个数，nc * nc 说明这个矩阵记录的是是簇与簇之间的关系 affinity measure between two clusters
    affinity = np.zeros([nc,nc])

    for i in range(nc):
        for j in range(i+1,nc):
            ij = np.ix_(Vc[i],Vc[j])    # ix_生成一个过滤器 ij，提取Vc[i]行 Vc[j]列的元素
            ji = np.ix_(Vc[j],Vc[i])

            W_ij, W_ji = W[ij], W[ji]
            Ci, Cj = len(Vc[i]),len(Vc[j])

            ones_i = np.ones((Ci,1))
            ones_j = np.ones((Cj,1))
            affinity[i][j] = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)
            affinity[j][i] = affinity[i][j]    # affinity 是一个对称的矩阵
    return affinity

def getAffinityBtwCluster(C1, C2, W):

    ij = np.ix_(C1, C2)
    ji = np.ix_(C2, C1)

    W_ij, W_ji = W[ij], W[ji]
    Ci, Cj = len(C1), len(C2)

    ones_i = np.ones((Ci, 1))
    ones_j = np.ones((Cj, 1))
    affinity = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)

    return affinity[0,0]

""" 
Vc : 二维列表，初始化的簇
Kc : K近邻
W : 权重矩阵
"""
def getNeighbor(Vc, Kc, W):

    Ns, As = [], []
    A = getAffinityMaxtrix(Vc, W)

    for i in range(len(A)):
        As.append([x for x in sorted(list(A[i]))[-1 * Kc:] if x > 0])
        n = len(As[i])
        if n==0:
            Ns.append([])
        else:
            Ns.append(A[i].argsort()[-1*n:].tolist())

    return Ns,As

# Ks : the number of neighbors for KNN graph
def AGDL(data, distance, targetClusterNum, Ks, Kc, a=1):

    print("data length : ", len(data))
    indices = np.argsort(distance, axis=1)[:, 1:Ks + 1]
    #distance = np.sort(distance, axis=1)[:, 1:Ks+1]
    #sorted_dist = np.sort(distance, axis=1)[:, 1:Ks+1]

    cluster = k0graph(distance, a)    # 初始化，把所有的数据点分为 len(cluster) 个簇，cluster是一个二维列表
    length = 0
    for i in range(len(cluster)):
        length += len(cluster[i])

    print("data before clustering : ", length)    # length应该等于data的数量

    # 生成权重矩阵W，每个点的 Ks 邻居之间是有权值的, 即 W[i] 行有 Ks 个不为0 的数
    #W,sigma2 = w_matrix(data, distance, indices, Ks, a)
    W, sigma2 = gacBuildDigraph(distance, Ks, a)

    neighborSet, affinitySet = getNeighbor(cluster, Kc, W)
    currentClusterNum = len(cluster)

    print("After k0 clustering : ", currentClusterNum)
    while currentClusterNum > targetClusterNum:
        max_affinity = 0
        max_index1 = 0
        max_index2 = 0
        for i in range(len(neighborSet)):
            if len(neighborSet[i])==0:
                continue
            aff = max(affinitySet[i])
            if aff > max_affinity:
                j = int(neighborSet[i][affinitySet[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            print("index alias")
            print(affinitySet)
            break

        #merge two cluster
        cluster[max_index1].extend(cluster[max_index2])
        cluster[max_index2] = []

        if max_index2 in neighborSet[max_index1]:
            p = neighborSet[max_index1].index(max_index2)
            del neighborSet[max_index1][p]
        if max_index1 in neighborSet[max_index2]:
            p = neighborSet[max_index2].index(max_index1)
            del neighborSet[max_index2][p]

        for i in range(len(neighborSet)):
            if i==max_index1 or i==max_index2:
                continue

            if max_index1 in neighborSet[i]:
                aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)

                p = neighborSet[i].index(max_index1)
                affinitySet[i][p] = aff_update # fix the affinity values

            if max_index2 in neighborSet[i]:
                p = neighborSet[i].index(max_index2)
                del neighborSet[i][p]
                del affinitySet[i][p]

                if max_index1 not in neighborSet[i]:
                    aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)
                    neighborSet[i].append(max_index1)
                    affinitySet[i].append(aff_update)

        neighborSet[max_index1].extend(neighborSet[max_index2])
        neighborSet[max_index1] = list(set(neighborSet[max_index1]))

        affinitySet[max_index1] = []

        neighborSet[max_index2] = []
        affinitySet[max_index2] = []

        # Fine the Kc-nearest clusters for Cab

        for i in range(len(neighborSet[max_index1])):
            target_index = neighborSet[max_index1][i]
            newAffinity = getAffinityBtwCluster(cluster[target_index], cluster[max_index1], W)
            affinitySet[max_index1].append(newAffinity)

        if len(affinitySet[max_index1]) > Kc:
            index = np.argsort(affinitySet[max_index1])
            new_neighbor = []
            new_affinity = []
            for j in range(Kc):
                new_neighbor.append(neighborSet[max_index1][index[-1*j]])
                new_affinity.append(affinitySet[max_index1][index[-1*j]])

            neighborSet[max_index1] = new_neighbor
            affinitySet[max_index1] = new_affinity

        currentClusterNum = currentClusterNum - 1


    reduced_cluster = []
    for i in range(len(cluster)):
        if len(cluster[i]) != 0:
            reduced_cluster.append(cluster[i])
    length = 0
    for i in range(len(reduced_cluster)):
        length += len(reduced_cluster[i])
    print("Data number, Cluster number : ", length, len(reduced_cluster))

    return reduced_cluster


