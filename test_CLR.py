# -*- coding:utf-8 -*-

import numpy as np
import ClusteringAlgorithm.CLR as CACLR

K = 5
groupNumber = 20

fea = np.loadtxt('fea.txt')
A0 = np.loadtxt('A0.txt')

print("debug")

cl, S, evs, cs = CACLR.CLR(A0, groupNumber, 0, 1)

print("end")