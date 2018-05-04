#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 05:19:16 2018

@author: teddy
"""

import pickle
import numpy as np
import os

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')  # 进行矩阵变换
        y = np.array(y)
        return x,y
    
def load_cifar10(root):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(root,'data_batch_%d' % (b,))
        print(f)
        x,y = load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
        del x,y    # del data immediatily
    Xtrain = np.concatenate(xs)            # 对xs进行拼接，即把三色的数据拼接到一起。
    Ytrain = np.concatenate(ys)
    
    Xtest, Ytest = load_cifar_batch(os.path.join(root,'test_batch'))
    return Xtrain,Ytrain,Xtest,Ytest
