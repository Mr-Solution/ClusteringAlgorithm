# -*- coding:utf-8 -*-
from . import rodfunc
import numpy as np
import matplotlib.pyplot as plt

jain = np.loadtxt('jain.txt')
features = jain[:,:2]
plt.scatter(features[:,0], features[:,1])
plt.show()

print('hello')
Clusters_lists = rodfunc.rod(features, 20, 10)

print()
for i in range(len(Clusters_lists)):
    print(Clusters_lists[i])