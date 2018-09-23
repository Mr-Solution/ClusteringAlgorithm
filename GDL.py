# -*- coding:utf-8 -*-

from tool import tool
import numpy as np
import copy

def gdl(dist_set, groupNumber, K=20, a=1, usingKcCluster=True, p=1):
    print("------ Building graph and forming iniital clusters with l-links ------")
