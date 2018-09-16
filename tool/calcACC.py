import numpy as np

from ClusteringAlgorithm.tool.hungarian import Hungarian

def calcACC(groundTruth, predValue):
    """
    计算使用预测值与真实值的最佳匹配的正确率
    :param groundTruth:真实值 np.ndarray
    :param predValue:预测值 np.ndarray
    :return:ACC:正确率 
    """
    if len(groundTruth.shape) != 1:
        groundTruth = groundTruth.reshape(groundTruth.shape[0])
    if len(predValue.shape) != 1:
        predValue = predValue.reshape(predValue.shape[0])
    # predValue 匹配真实值 groundTruth
    res = bestMap(groundTruth, predValue)
    ACC = np.sum(res == groundTruth) / groundTruth.shape[0]
    return ACC


def bestMap(L1, L2):
    """
    两个向量的最佳匹配
    :param L1:np.ndarray
    :param L2:np.ndarray
    :return:L1对L2的最佳匹配 np.ndarray
    """
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum(((L1 == Label1[i]) & (L2 == Label2[j])))
    hungarian = Hungarian(G, is_profit_matrix=True)
    hungarian.calculate()
    resultMap = hungarian.get_results()
    resultMap = sorted(resultMap, key=lambda s: s[1]) # resultMap[1] = (trueId, predId)
    newL = np.zeros(L1.shape[0], dtype=np.uint)
    for i,v in enumerate(Label1):
        newL[L2 == v] = Label1[resultMap[i][0]]
    return newL