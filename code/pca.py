#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/7/18 10:18
# @File    : pca.py

import matplotlib.pyplot as plt                 # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           # 加载PCA算法包
from sklearn.datasets import load_iris


# 将鸢尾花数据集的四维数据降至二维
def four_two():
    data = load_iris()
    y = data.target
    x = data.data
    pca = PCA(n_components=2)         # 加载PCA算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(x)  # 对样本进行降维

    red_x, red_y = [],[]
    blue_x, blue_y = [],[]
    green_x, green_y = [],[]

    for i in range(len(reduced_x)):
        if y[i] ==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])

        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])

        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    # 可视化
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    plt.show()


# 将鸢尾花数据集的前两维降至一维
def two_one():
    data = load_iris()
    x_2D = data.data[:, :2]
    y = data.target
    pca = PCA(n_components=1)  # 加载PCA算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(x_2D)  # 对样本进行降维

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for i in range(len(reduced_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(y[i])

        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(y[i])

        else:
            green_x.append(reduced_x[i][0])
            green_y.append(y[i])

    # 可视化
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


if __name__ == "__main__":
    four_two()

