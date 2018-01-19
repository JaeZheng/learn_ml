import numpy as np
import matplotlib.pyplot as plt


# sigmod函数，即分类的函数
def sigmod(x):
    return 1/(1+np.exp(-x))


# 代价函数
def cost(hx, y):
    return -y * np.log(hx) - (1-y) * np.log(1-hx)


# 梯度下降函数
def gradient(current_para, x, y, learning_rate):
    m = len(y)
    matrix_gradient = np.zeros(len(x[0]))
    for i in range(m):
        current_x = x[i]
        current_y = y[i]
        current_x = np.asarray(current_x)
        matrix_gradient += (sigmod(np.dot(current_para, current_x)) - current_y) * current_x

    new_para = current_para - learning_rate * matrix_gradient
    return new_para


# 误差计算函数
def error(para, x, y):
    total = len(y)
    error_num = 0
    for i in range(total):
        current_x = x[i]
        current_y = y[i]
        hx = sigmod(np.dot(para, current_x))
        if cost(hx, current_y) > 0.5:
            error_num += 1

    return error_num/total


# 训练过程
def train(initial_para, x, y, learning_rate, num_iter):
    dataMat = np.asarray(x)
    labelMat = np.asarray(y)
    para = initial_para
    for i in range(num_iter+1):
        para = gradient(para, dataMat, labelMat, learning_rate)
        if i % 100 == 0:
            err = error(para, dataMat, labelMat)
            print("iter:" + str(i) + " ; error:" + str(err))

    return para


# 加载数据集
def load_dataset():
    dataMat = []
    labelMat = []
    with open("../data/logistic_regression_binary.csv", "r+") as file_object:
        lines = file_object.readlines()
        for line in lines:
            line_array = line.strip().split()
            # 数据矩阵
            dataMat.append([1.0, float(line_array[0]), float(line_array[1])])
            # 标签矩阵
            labelMat.append(int(line_array[2]))

    return dataMat, labelMat


def plotBestFit(wei, data, label):
    if type(wei).__name__ == 'ndarray':
        weights = wei
    else:
        weights = wei.getA()
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    xxx = np.arange(-3,3,0.1)
    yyy = - weights[0]/weights[2] - weights[1]/weights[2]*xxx
    ax.plot(xxx,yyy)
    cord1 = []
    cord0 = []
    for i in range(len(label)):
        if label[i] == 1:
            cord1.append(data[i][1:3])
        else:
            cord0.append(data[i][1:3])
    cord1 = np.array(cord1)
    cord0 = np.array(cord0)
    ax.scatter(cord1[:, 0], cord1[:, 1], c='red')
    ax.scatter(cord0[:, 0], cord0[:, 1], c='green')
    plt.show()


def logistic_regression():
    x, y = load_dataset()
    n = len(x[0])
    initial_para = np.ones(n)
    learning_rate = 0.001
    num_iter = 1000
    print("起始参数：")
    print(initial_para)
    para = train(initial_para, x, y, learning_rate, num_iter)
    print("最后训练得到的参数：")
    print(para)
    plotBestFit(para, x, y)


logistic_regression()

