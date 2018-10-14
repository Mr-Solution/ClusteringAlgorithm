# -*- coding:utf-8 -*-

import numpy as np
from PIL import Image
import os
import glob    # python 自带的文件操作模块，支持通配符 *.jpg 操作
import time

def load_coil100(data_path='dataset/coil-100'):
    print("load COIL100...... Path :",data_path)

    file_list = os.listdir(data_path)
    #print(file_list)
    dataSet = []
    labels = []
    for file in file_list:
        label = int(file.split('__')[0][3:])
        labels.append(label)
        file = os.path.join(data_path, file)
        #print(file)
        data = []
        im = Image.open(file)
        pix = im.load()
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                #r,g,b = pix[x,y]
                #print(pix[x,y])
                data.extend(pix[x,y])
        dataSet.append(data)

    return dataSet, labels


def load_coil100_2(data_path):
    print("load COIL100")
    print("path :",data_path)

    dataSet = []
    labels = []
    for img_file in glob.glob(data_path+'/*.png'):
        label = int(img_file.split('\\')[-1][3])
        labels.append(label)
        im = Image.open(img_file)
        pix = im.load()
        data = []
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                data.extend(pix[x, y])

        dataSet.append(data)

    # file_list = os.listdir(data_path)
    # print(file_list)
    # dataSet = []
    # labels = []
    # for file in file_list:
    #     label = int(file.split('__')[0][3:])
    #     labels.append(label)
    #     file = os.path.join(data_path, file)
    #     #print(file)
    #     data = []
    #     im = Image.open(file)
    #     pix = im.load()
    #     for x in range(im.size[0]):
    #         for y in range(im.size[1]):
    #             #r,g,b = pix[x,y]
    #             #print(pix[x,y])
    #             data.extend(pix[x,y])
    #     dataSet.append(data)

    return dataSet, labels

if __name__ == '__main__':
    start = time.time()
    #dataSet, labels = load_coil100('D:/WorkSpace/GitHub/ClusteringAlgorithm/dataset/coil-100')
    dataSet, labels = load_coil100()
    end = time.time()
    print(end - start)
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    end2 = time.time()
    print(end2 - end)
    print(dataSet.shape)
    print(labels)