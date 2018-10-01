"""
# -*- coding:utf-8 -*-

import numpy as np
from scipy.spatial import distance
#from scipy.sparse import coo_matrix
import scipy.sparse
from sklearn.cluster import KMeans

def predict_ncut(fea, nClass):
    W = compute_relation(fea)
    NcutDiscrete = ncutW(W, nClass)
    label_pre = np.zeros(fea.shape[0])
    for i in range(nClass):
        label_pre[np.where(NcutDiscrete[:,i] == 1)] = i

    return label_pre

"""
# data: Num_data X feature_dim
# return W : pair-wise data similarity matrix
"""
def compute_relation(data, order=2):
    distances = np.square(distance.cdist(data, data))
    scale_sig = 0.05 * int(np.max(distances))    # np.max()计算结果中出现了10.000000000000002这样的小数
    tmp = np.power(distances/scale_sig, order)
    return np.exp(-tmp)

def ncutW(W, nClass):
    ncut(W, nClass)


def ncut(W, nClass=8):
    dataNcut = {'offset':5e-1,
                'verbose':0,
                'maxiterations':300,
                'eigsErrorTolerance':1e-8,
                'valeurMin':1e-6}

    # sparsify the matrix W, ignore the values of matrix which are below a threshold
    W = scipy.sparse.coo_matrix(np.where(W>dataNcut['valeurMin'], W, 0))

    if np.sum(np.abs(W - W.T)) > 1e-10:
        print("err! W not symmetric")
        return -1

    n = W.shape[0]
    nClass = min(nClass, n)
    offset = dataNcut['offset']

    # degrees and regularization
    d = np.sum(np.abs(W), axis=1)
    dr = 0.5 * (d - np.sum(W,axis=1))
    d = d + offset * 2
    dr = dr + offset
    W = W + scipy.sparse.spdiags(dr,0,n,n)

    Dinvsqrt = 1/np.sqrt(d+2.2204e-16)
    # function spmtimesd
    P = np.dot(np.diag(Dinvsqrt), W)
    P = np.dot(P, np.diag(Dinvsqrt))

    options = {
        'issym':1,
        'disp':0,
        'maxit':dataNcut['maxiterations'],
        'tol':dataNcut['eigsErrorTolerance'],
        'v0':np.ones(P.shape[0]),
        'p':min(n, max(35,2*nClass)),
    }
"""

# encoding=utf-8
import numpy as np
import tool.tool as tool
from scipy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from sklearn import metrics

def similarity_function(points):
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def normalized_cut(points, k):
    A = similarity_function(points)
    W = np.eye(len(A)) - normalize(A, norm='l1')
    eigvalues, eigvectors = LA.eig(W)
    indices = np.argsort(eigvalues)[1:k]
    return KMeans(n_clusters=k).fit_predict(eigvectors[:, indices])


data = np.loadtxt('dataset/COIL20_32.txt')
X = data[:, :-1]
y = data[:, -1]
X = tool.data_Normalized(X)
groupNumber = len(np.unique(y))

# X, y = make_blobs()
labels = normalized_cut(X, groupNumber)

NMI = metrics.adjusted_mutual_info_score(y, labels)
print(NMI)