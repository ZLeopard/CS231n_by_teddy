#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 23:52:50 2018

@author: teddy
"""

import numpy as np

class KNearestNeighbor:
    
    def __init__(self):
        pass
    
    def train(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def predict(self,X,k=1,num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loop(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists,k=k)
    
    def compute_distances_two_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))   # 创建零矩阵
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.sum((X[i,:]-self.X_train[j,:])**2))
        return dists
    
    def compute_distances_one_loop(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1)) #sum in the raw
        return dists
    
    def compute_distances_no_loop(self,X):   # a sample is a raw array!
        num_test = X.shape[0]    # the number of sample
        num_train = self.X_train.shape[0]   # the number of columns
        dists = np.zeros((num_test, num_train))
        test_sum = np.sum(np.square(X),axis=1)   # sum in the raw
        train_sum = np.sum(np.square(self.X_train),axis=1)
        inner_product = np.dot(X,self.X_train.T)  # inner_product
        dists = np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)   # -1 is all
        return dists
    
    def predict_labels(self,dists,k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            y_indicies = np.argsort(dists[i,:],axis=0)  # argsort return the index from low to high
            closest_y = self.y_train[y_indicies[:k]]    # find k nearest vector's indecies
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred