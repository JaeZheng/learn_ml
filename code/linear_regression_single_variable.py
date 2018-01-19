# y = mx + b
import numpy as np
import matplotlib.pyplot as plt


# 误差计算函数
def compute_error(m, b, data):
    x = data[:, 0]
    y = data[:, 1]
    N = len(x)
    total_error = (y-m*x-b)**2
    total_error = np.sum(total_error, axis=0)
    return total_error/(2*N)


# 梯度下降函数
def compute_gradient(m_current, b_current, data, learning_rate):
    m_gradient = 0
    b_gradient = 0

    X = data[:, 0]
    N = len(X)

    for i in range(0, N):
        x = data[i, 0]
        y = data[i, 1]
        m_gradient += (m_current*x+b_current-y)*x
        b_gradient +=  m_current*x+b_current-y

    m_new = m_current-learning_rate*m_gradient/N
    b_new = b_current-learning_rate*b_gradient/N

    return [m_new, b_new]


# 参数优化过程，输出误差
def optimizer(m_init, b_init, data, learing_rate, num_iter):
    m = m_init
    b = b_init

    for i in range(0, num_iter+1):
        m, b = compute_gradient(m, b, data, learing_rate)
        if i % 100 == 0:
            error = compute_error(m, b, data)
            print("iter:"+str(i)+" ; error:"+str(error))

    return [m, b]


# 绘图函数
def plot_show(m, b, data):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='blue')
    plt.plot(x, m*x+b, color='red', linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# 正则化
def regularize(data):#regularize by columns
    x = data[:, 0]
    y = data[:, 1]
    data_tmp = data.copy()
    x_scale = x.max() - x.min()
    x_mean = x.mean()
    y_scale = y.max() - y.min()
    y_mean = y.mean()
    x_new = (x-x_mean)/x_scale
    y_new = (y-y_mean)/y_scale
    data_tmp[:, 0] = x_new
    data_tmp[:, 1] = y_new
    return data_tmp


def linear_regression():
    data = np.loadtxt('../data/linear_regression_single_variable.csv', delimiter=',')
    # data = regularize(data)
    learning_rate = 0.01
    initial_b = 0.0
    initial_m = 0.0
    num_iter = 2500

    print("单变量线性回归：")
    print("学习率："+str(learning_rate))
    print("起始m："+str(initial_m))
    print("起始b："+str(initial_b))
    print("迭代次数："+str(num_iter))

    [m, b] = optimizer(initial_m, initial_b, data, learning_rate, num_iter)

    print("最后训练出来的参数：")
    print("m："+str(m))
    print("b："+str(b))

    plot_show(m, b, data)


linear_regression()
