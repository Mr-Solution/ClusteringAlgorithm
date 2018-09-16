# -*- coding:utf-8 -*-
import os
import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


"""
Data Struct
"""
counter = 0
absolute_distances = []
nn_indices = []
nn_distances = []
pts_distances = []
Clusters_lists = []
clu_num = 0
clu_num_old = 0


"""
Distance Function
"""

"""
 return the order of Cb in Ca's order list
 Ca and Cb are the cluster_IDs 
"""
def Order(Ca, Cb) :
    global nn_indices
    # np.where return an array [[i]], use [0][0] to get the value i
    return np.where(nn_indices[Ca] == Cb)[0][0]  


"""
 Rank-Order distance between Ca and Cb        formula (1)
"""
def Dist(Ca, Cb) :
    global nn_indices
    dis = 0
    for i in range(Order(Ca, Cb)) :
        dis = dis + Order(Cb, nn_indices[Ca][i])

    return dis

"""
 the closest distance between two clusters    formula (3)
 Ca and Cb are cluster_IDs
"""
def dist(Ca, Cb) :
    global absolute_distances
    return absolute_distances[Ca][Cb]


"""
 DR(Ca, Cb)                                    formula (4)
"""
def Dist_R(Ca, Cb):
    return (Dist(Ca, Cb) + Dist(Cb, Ca))/min(Order(Ca, Cb), Order(Cb, Ca))

"""
 DN(Ca, Cb)                                    formula (5)
 Ca, Cb is the index of pts in features
"""
def Dist_N(Ca, Cb, K=3):
    """ K is a constant parameter """
    global pts_distances
    pts_Ca = Clusters_lists[Ca]
    pts_Cb = Clusters_lists[Cb]
    phi = (np.sum(pts_distances[pts_Ca,1:K+1]) + np.sum(pts_distances[pts_Cb,1:K+1]))/ \
    (K * (len(pts_Ca) + len(pts_Cb)))
    return dist(Ca, Cb) / phi


"""
return merged candidates
"""
def merge_candidates(candidates):
    merged_candidates = []
    for candidate in candidates:
        if len(merged_candidates) == 0:
            merged_candidates.append(candidate)
        else:
            c = set(candidate)
            i = 0
            while i<len(merged_candidates):
                mc = set(merged_candidates[i])
                if not c.isdisjoint(mc):
                    merged_candidates[i] = merged_candidates[i] + list(c - mc)
                    break
                i = i+1
            if i==len(merged_candidates):
                merged_candidates.append(candidate)
                    
    merged_candidates_II = []
    for candidate in merged_candidates:
        if len(merged_candidates_II) == 0:
            merged_candidates_II.append(candidate)
        else:
            c = set(candidate)
            i = 0
            while i<len(merged_candidates_II):
                mc = set(merged_candidates_II[i])
                if not c.isdisjoint(mc):
                    merged_candidates_II[i] = merged_candidates_II[i] + list(c - mc)
                    break
                i = i+1
            if i==len(merged_candidates_II):
                merged_candidates_II.append(candidate)

    return merged_candidates_II


def merge_cluster_lists(candidates) :
    print('candidates :',candidates)
    global Clusters_lists
    # candidate is also a list
    for candidate in candidates:
        candidate.sort()
        for i in range(1, len(candidate)):
            Clusters_lists[candidate[0]] = Clusters_lists[candidate[0]] + Clusters_lists[candidate[i]]
            #Clusters_lists[candidate[0]].append(candidate[i])
            # append()会把list整体加入原来list [1,2,3,[4,5,6]]
            # 我们需要的效果是[1,2,3,4,5,6]

    del_list = []
    for candidate in candidates:
        del_list = del_list + candidate[1:]
    # 删除列表元素的时候列表长度会发生变化，所以要倒序从尾部删除
    del_list.sort(reverse=True)
    print('del_list :',del_list)
    print('len(Clusters_lists) :',len(Clusters_lists))
    for index in del_list:
        Clusters_lists.pop(index)

"""
 candidates: a list of cluster-pairs which are candidate merging clusters
"""
def transitive_merge(candidates) :
    global clu_num
    global clu_num_old
    global counter
    global absolute_distances
    global nn_indices
    global nn_distances
    # merge the pairs in candidate list
    candidates = merge_candidates(candidates)
    #print("merged_candidates =", candidates)
    # merge the clusters lists
    merge_cluster_lists(candidates)
    #print("candidates =", candidates)
    # update clu_num
    clu_num_old = clu_num
    clu_num = len(Clusters_lists)
    # merge absolute_distances
    sl = []
    for candidate in candidates:
        new_index = min(candidate)
        old_distance = absolute_distances[candidate]
        new_distance = old_distance.min(axis = 0)
        absolute_distances[new_index] = new_distance
        absolute_distances[:, new_index] = absolute_distances[new_index].T
        #candidates[candidates.index(candidate)].remove(new_index)
        candidate.remove(new_index)
        sl = sl + candidate

    # qu chong
    sl = set(sl)
    sl = list(sl)

    absolute_distances = np.delete(absolute_distances, sl, axis = 0) # shan chu hang
    absolute_distances = np.delete(absolute_distances, sl, axis = 1) # shan chu lie
    # update nn_indices
    nn_indices = np.argsort(absolute_distances)
    nn_distances = np.sort(absolute_distances)
    # counter
#    print("merge : %d  clu_num = %d  clu_num_old = %d" % (counter, clu_num, clu_num_old))
    counter = counter + 1


def rod(features, t=20, K = 5):
    global absolute_distances
    global nn_indices
    global nn_distances
    global pts_distances
    global Clusters_lists
    global clu_num
    global clu_num_old

    n_samples = features.shape[0]
    for i in range(n_samples):
        Clusters_lists.append([i])
    
    absolute_distances = distance.cdist(features, features, 'euclidean')
    nn_indices = np.argsort(absolute_distances)
    nn_distances = np.sort(absolute_distances)
    #pts_distances = copy.deepcopy(absolute_distances)
    pts_distances = copy.deepcopy(nn_distances)

    clu_distances = absolute_distances
    clu_order_lists = nn_indices

    clu_num = len(Clusters_lists)
    clu_num_old = 0;
    
    #print('orgin Dist_R & Dist_N :')
    #for i in range(n_samples):
    #    for j in range(i+1, n_samples):
    #        print('Dist_R(%d,%d) = %.2f    Dist_N(%d,%d) = %.2f'%(i,j,Dist_R(i,j), i,j,Dist_N(i,j)))
    #
    while True:
        print()
        print('counter = %d  clu_num = %d  clu_num_old = %d'%(counter,clu_num,clu_num_old))
        candidates = []
        # repeat until no merge happens
        if clu_num == clu_num_old :
            break;
        for Ci in range(0,clu_num):
            for Cj in range(Ci+1,clu_num):
                Rank_Order_dist = Dist_R(Ci, Cj)
                Normalized_dist = Dist_N(Ci, Cj, K)
                print('Dist_R(%d,%d) = %.2f    Dist_N(%d,%d) = %.2f'%(Ci,Cj, Rank_Order_dist, Ci,Cj, Normalized_dist))              
                
                if Rank_Order_dist < t and Normalized_dist < 1.5:
                    candidates.append([Ci,Cj])

        #print('candidates :',candidates)
        if len(candidates) > 0 :
            transitive_merge(candidates)
        else :
            break

    print()
    print('Clusters_list :',Clusters_lists)

    return Clusters_lists