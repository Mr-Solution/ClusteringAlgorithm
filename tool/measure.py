from sklearn import metrics
# from munkres import Munkres, make_cost_matrix
import numpy as np
from . import hungarian

def NMI(labels_real, labels_pred):
    nmi = metrics.normalized_mutual_info_score(labels_real, labels_pred)
    return nmi

def ACC(groundTruth, predValue):
    """
    计算使用预测值与真实值的最佳匹配的正确率
    :param groundTruth:真实值 np.ndarray
    :param predValue:预测值 np.ndarray
    :return:ACC:正确率
    """
    if len(np.unique(groundTruth)) != len(np.unique(predValue)):
        print("ERROR! Number of labels_pred should be equal with the number of labels.")
        return -1

    if len(groundTruth.shape) != 1:
        groundTruth = groundTruth.reshape(groundTruth.shape[0])
    if len(predValue.shape) != 1:
        predValue = predValue.reshape(predValue.shape[0])

    # predValue 匹配真实值 groundTruth
    res = bestMap(groundTruth, predValue)
    ACC = np.sum(res == groundTruth) / groundTruth.shape[0]
    return ACC


def bestMap(labels_real, labels_pred):
    """
    两个向量的最佳匹配
    :param labels_real:np.ndarray
    :param labels_pred:np.ndarray
    :return:L1对L2的最佳匹配 np.ndarray
    """
    Label1 = np.unique(labels_real)
    nClass1 = len(Label1)
    Label2 = np.unique(labels_pred)
    nClass2 = len(Label2)

    # predValue中间有跳过的label ： 0，1，5，6，10
    if nClass2 < np.max(labels_pred):
        newLabel2 = np.argsort(Label2)
        for i in range(nClass2):
            if Label2[i] != newLabel2[i]:
                labels_pred = np.where(labels_pred == Label2[i], newLabel2[i], labels_pred)
        Label2 = newLabel2

    # 如果groundTruth是从1开始计数，而predValue是从0开始计数，那么predValue += 1
    if nClass1 == np.max(labels_real) and nClass2 > np.max(labels_pred):
        labels_pred = labels_pred + 1
        Label2 = Label2 + 1

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum(((labels_real == Label1[i]) & (labels_pred == Label2[j])))
    myhungarian = hungarian.Hungarian(G, is_profit_matrix=True)
    myhungarian.calculate()
    resultMap = myhungarian.get_results()
    resultMap = sorted(resultMap, key=lambda s: s[1])  # resultMap[1] = (trueId, predId)
    newL = np.zeros(labels_real.shape[0], dtype=np.uint)
    for i,v in enumerate(Label1):
        newL[labels_pred == v] = Label1[resultMap[i][0]]
    return newL
