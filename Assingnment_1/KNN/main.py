#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 05:33:19 2018

@author: teddy
"""

import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from KNN import KNearestNeighbor
import time

def time_function(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

x_train,y_train,x_test,y_test = load_cifar10('../cifar-10-batches-py')

print('training data shape:', x_train.shape)
print('training labels shape:', y_train.shape)
print('test data shape:', x_test.shape)
print('test labels shape:', y_test.shape)

# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7    #没个类选择7个样本
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train==y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i*num_classes+y+1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(x_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

num_training = 5000    # 选取5000个训练样本，500个测试集
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]

# 把数据变成行向量
x_train = np.reshape(x_train,(x_train.shape[0],-1))
x_test = np.reshape(x_test, (x_test.shape[0],-1))
print(x_train.shape, x_test.shape)     # 把原来的数据（32,32,3）转化成（3072，）即32*32*3=3072

# 测试集预测
classifier = KNearestNeighbor()     # KNN分类器
classifier.train(x_train,y_train)
dists = classifier.compute_distances_no_loop(x_test)   # 计算样本间的距离

# 分析三个函数运行时间的差距
# two_loop_time = time_function(classifier.compute_distances_two_loops, x_test)
# print('two loops version took %f seconds' % two_loop_time)
#
# one_loop_time = time_function(classifier.compute_distances_one_loop, x_test)
# print('one loop version took %f seconds' % one_loop_time)
#
# no_loop_time = time_function(classifier.compute_distances_no_loop, x_test)
# print('no loops version took %f seconds' % no_loop_time)

# print(dists)   # 输出距离矩阵，不过数据量太大了
y_test_pred = classifier.predict_labels(dists, k=1)    # k=1为最近邻情况
num_correct = np.sum(y_test_pred == y_test)            # 正确的预测情况
accuracy = float(num_correct) / num_test
print("got %d / %d correct => accuracy: %f" % (num_correct, num_test, accuracy))

# 交叉验证来选择最好的k值来获取较好的K值（要根据最后的精度加以确定）
num_folds = 5    # 把数据分为5份
k_choices = [1,3,5,8,10,12,15,20,50,100]
x_train_folds = []
y_train_folds = []

y_train = y_train.reshape((-1,1))      # 变为列向量
x_train_folds = np.array_split(x_train, num_folds)  # 把x_train矩阵分成5份
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies.setdefault(k, [])     # 设置初值
for i in range(num_folds):
    classifier = KNearestNeighbor()
    x_val_train = np.vstack(x_train_folds[0:i]+x_train_folds[i+1:])
    y_val_train = np.vstack(y_train_folds[0:i]+y_train_folds[i+1:])     # 选取第i份为测试集，其余为训练集
    y_val_train = y_val_train[:,0]   # 转置为行向量，相当于y_val_train.T
    classifier.train(x_val_train, y_val_train)
    for k in k_choices:
        dists = classifier.compute_distances_no_loop(x_train_folds[i])
        y_val_pred = classifier.predict_labels(dists, k=k)   # 第i份为测试集进行测试
        num_correct = np.sum(y_val_pred == y_train_folds[i][:,0])       # y_train_folds[i]为列向量，转置后面加[:,0] 相当于.T
        accuracy = float(num_correct) / len(y_val_pred)                 # 在测试集计算精度
        k_to_accuracies[k] += [accuracy]                                # k_to_accuracies[k].append(accuracy)

for k  in sorted(k_to_accuracies):           # 交叉验证后进行求取平均值
    sum_accuracy = 0
    for accuracy in k_to_accuracies[k]:
        print('k=%d, accuracy=%f' % (k, accuracy))
        sum_accuracy += accuracy
    print('the average accuracy is :%f' % (sum_accuracy/5))    # 最后的平均精度

# **************         通过计算的K=10的时候精度最好            ***************
# 绘制图形用来显示不同k值精度的不同
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k]*len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for v in sorted(k_to_accuracies.values())])   # 计算不同k值的均值
accuracies_std = np.array([np.std(v) for v in sorted(k_to_accuracies.values())])     # 计算标准差

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('cross-validation on k')
plt.xlabel('k')
plt.ylabel('cross-validation accuracy')
plt.show()