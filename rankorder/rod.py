# -*- coding:utf-8 -*-
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import fileinfo
import copy
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

counter = 0

pic_dir_list = os.listdir(fileinfo.TRAIN_FILE_DIR)
m = len(pic_dir_list)
print(m)
fig = plt.figure()

imgs = np.arange(92*112)
for i in range(10):
    sub_dir = pic_dir_list[i]
    pic_dir = os.path.join(fileinfo.TRAIN_FILE_DIR, sub_dir)
    print(pic_dir)
    if (os.path.isdir(pic_dir)):
        piclist = os.listdir(pic_dir)
        print(piclist)
        for j in range(10):
            pic = piclist[j]
            pic = os.path.join(pic_dir,pic)
            testimg = mpimg.imread(pic)
            plt.subplot(10,10, 10*i+j+1)
            plt.imshow(testimg)
            plt.axis('off')
            img_vec = np.asarray(testimg)
            # print(img_vec)
            img_vec = np.reshape(img_vec,(1,92*112))
            # print(img_vec)
            imgs = np.vstack((imgs, img_vec))

fig.tight_layout()
plt.subplots_adjust(wspace =0, hspace =0)
# plt.show()


"""
Data Struct
absolute_distances : distances between samples and samples, is a nxn matrix
nn_indices : n-nearest neighbors order lists
nn_distances : n-nearesr neighbors distances lists
Clusters_lists : store the clustering result
"""

# At line 16, imgs was inited as np.arange(92*112) 
imgs = imgs[1:,:]
n_samples = imgs.shape[0]
pca = decomposition.PCA(n_components=6, svd_solver='randomized', whiten=True)
# use pca method to extract features
features = pca.fit_transform(imgs)
#np.savetxt("features.txt", features)

# absolute_pts_distances matrix
absolute_distances = distance.cdist(features, features, 'euclidean')
# generate order lists accroding to absolute distance
nn_indices = np.argsort(absolute_distances)
nn_distances = np.sort(absolute_distances)
pts_distances = copy.deepcopy(nn_distances)

# nbrs = NearestNeighbors(n_neighbors=n_samples, algorithm='ball_tree').fit(features)
# nn_distances, nn_indices = nbrs.kneighbors(features)

# Clusters_lists : [[C1],[C2],[C3],...,[Cn]]  Ci is a list of ptID
Clusters_lists = []
for i in range(n_samples):
    Clusters_lists.append([i])

clu_distances = absolute_distances
clu_order_lists = nn_indices

print("absolute_distances :")
print(absolute_distances)
print("nn_indices :")
print(nn_indices)
# print("clu_distances.shape : %d %d" % clu_distances.shape)

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
 need to speed up
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
    (K * (len(Clusters_lists[Ca]) + len(Clusters_lists[Cb])))
    return dist(Ca, Cb) / phi


"""
return merged candidates
"""
def merge_candidates(candidates):
    merged_candidates = []
    while len(candidates) > 0:
        merge = False
        s = set(candidates[0])
        del_list = [0]
        for i in range(1, len(candidates)) :
            sp = set(candidates[i])
            if not s.isdisjoint(sp):
                merge = True
                s = s.union(sp)
                del_list.append(i)
                
        for j in range(len(candidates)-1, 0, -1):
            sp = set(candidates[j])
            if not s.isdisjoint(sp):
                merge = True
                s = s.union(sp)
                del_list.append(j)
            
        del_list = set(del_list)
        del_list = list(del_list)
#        print(del_list)

        if merge:
            merged_candidates.append(list(s))
        for j in reversed(del_list):
            candidates.pop(j)

#         print(merged_candidates)
#         print(candidates)
    return merged_candidates


def merge_cluster_lists(candidates) :
    global Clusters_lists
    # candidate is also a list
    for candidate in candidates:
        candidate.sort()
        for i in range(1, len(candidate)):
            Clusters_lists[candidate[0]] = Clusters_lists[candidate[0]] + Clusters_lists[candidate[i]]
        #candidates[candidates.index(candidate)] = candidate[1:]

    del_list = []
    for candidate in candidates:
        del_list = del_list + candidate[1:]
        #for index in candidate:
        #    del_list.append(index)
    # 删除列表元素的时候列表长度会发生变化，所以要倒序从尾部删除
    del_list.sort(reverse=True)
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
    print("merged_candidates =", candidates)
    # merge the clusters lists
    merge_cluster_lists(candidates)
    print("Clusters_lists =", Clusters_lists)
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
    print("merge : %d  clu_num = %d  clu_num_old = %d" % (counter, clu_num, clu_num_old))
    counter = counter + 1
#    print("absolute_distances :")
#    print(absolute_distances)
#    print("nn_indices :")
#    print(nn_indices)


# threshold t
t = 20
clu_num = len(Clusters_lists)
clu_num_old = 0;
while True:
    candidates = []
    # repeat until no merge happens
    if clu_num == clu_num_old :
        break;
    for Ci in range(0,clu_num):
        for Cj in range(Ci+1,clu_num):
            Rank_Order_dist = Dist_R(Ci, Cj)
            Normalized_dist = Dist_N(Ci, Cj)
            #print("ROD(%d,%d)=%d  Normalized_dist = %d"%(Ci,Cj,Rank_Order_dist,Normalized_dist))
            if Rank_Order_dist < t and Normalized_dist < 1:
                candidates.append([Ci,Cj])

    if len(candidates) > 0 :
        transitive_merge(candidates)
    else :
        break
print(Clusters_lists)
#np.savetxt("Clusters_lists.txt", Clusters_lists)