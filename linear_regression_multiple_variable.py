import numpy as np


# 正则化
def regularize(x):#regularize by columns
    x_tmp = x.copy()

    for i in range(2, len(x[0])):
        current_x = x_tmp[:, i]
        x_avg = np.mean(current_x)
        x_var = current_x.max() - current_x.min()
        current_x = (current_x-x_avg)/(x_var)
        x_tmp[:, i] = current_x
    return x_tmp


# 梯度下降函数
def compute_gradient(current_para, x, y, learning_rate):
    length = len(x)
    matrix_gradient = np.zeros(len(x[0]))
    m = len(y)

    for i in range(0, length):
        current_x = x[i]
        current_y = y[i]
        current_x = np.asarray(current_x)
        matrix_gradient += (np.dot(current_para, current_x)-current_y)*current_x

    para_new = current_para - learning_rate*matrix_gradient/m
    return para_new


# 误差计算函数
def compute_error(para, x, y):
    total_error = 0
    length = len(x)
    num = len(y)
    for i in range(0, length):
        total_error += (y - (np.dot(para, x[i]))) ** 2
    total_error = np.sum(total_error, axis=0)
    return total_error / (2*num)


# 参数优化过程，输出误差
def optimizer(initial_para, x, y, learning_rate, num_iter):
    para = initial_para

    for i in range(0, num_iter+1):
        para = compute_gradient(para, x, y, learning_rate)
        if i % 100 == 0:
            error = compute_error(para, x, y)
            print("iter:"+str(i)+" ; error:"+str(error))

    return para


def linear_regression():
    data = np.loadtxt('linear_regression_multiple_variable.csv', delimiter='\t')
    # 特征数量
    n = len(data[0])
    # 样本数量
    m = len(data)
    x = data[:, 0:n-1]
    y = data[:, n-1]
    matrix_one = np.ones(m)
    x = np.insert(x, 0, values=matrix_one, axis=1)
    x = regularize(x)
    initial_para = np.zeros(n)
    print("起始参数：")
    print(initial_para)
    learning_rate = 0.001
    num_iter = 3400
    para = optimizer(initial_para, x, y, learning_rate, num_iter)
    print("最后训练得到的参数：")
    print(para)


linear_regression()