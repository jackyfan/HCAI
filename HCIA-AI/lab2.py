import numpy as np
import matplotlib.pyplot as plt
from bokeh.colors.groups import white
from holoviews.plotting.bokeh.styles import alpha


# 计算梯度
def generate_gradient(X, y, theta):
    sample_count = X.shape[0]
    # 计算梯度 采用矩阵运算 1/m ∑(((h(x^i)-y^i)) x_j^i)
    return (1.0 / sample_count) * X.T.dot(X.dot(theta) - y)


# 读取数据集
def get_training_data(file_path):
    original_data = np.loadtxt(file_path, skiprows=1)
    cols = original_data.shape[1]
    return original_data, original_data[:, :cols - 1], original_data[:, cols - 1:]


# 初始化参数
def init_theta(feature_count):
    # 初始化参数为全1
    return np.ones(feature_count).reshape(feature_count, 1)


def gradient_descending(X, y, theta, step):
    Jthetas = []
    index = 0
    gradient = generate_gradient(X, y, theta)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - step * gradient
        # 重新计算梯度
        gradient = generate_gradient(X, y, theta)
        # 计算损失函数，等于真实值与预测值的差的平方 (y^i-h(x^i)) ^2
        jTheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
        if (index + 1) % 10 == 0:
            Jthetas.append((index, jTheta[0]))
        index += 1
    return theta, Jthetas


# 显示损失函数曲线
def showJThetas(diff_value):
    p_x = []
    p_y = []
    for index, value in diff_value:
        p_x.append(index)
        p_y.append(value)
    plt.plot(p_x, p_y, color='b')
    plt.xlabel('steps')
    plt.ylabel('loss function')
    plt.title('step-loss function curve')
    plt.show()


# 显示数据点和拟合曲线
def showlinercurve(theta, sample_training_set):
    x, y = sample_training_set[:, 1], sample_training_set[:, 2]
    z = theta[0] + theta[1] * x
    plt.scatter(x, y, color='b', marker='x', label='sample data')
    plt.plot(x, z, 'r', label='regression curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('liner regression curve')
    plt.legend()
    plt.show()


def main():
    training_data_include_y, training_x, y = get_training_data('../dataset/ML/02/lr2_data.txt')
    sample_count, feature_count = training_x.shape
    # 定义学习步长
    step = 0.01
    # 初始化 theta
    theta = init_theta(feature_count)
    result_theta, Jthetas = gradient_descending(training_x, y, theta, step)

    print("w:{}".format(result_theta[0][0]), "b{}".format(result_theta[1][0]))

    showJThetas(Jthetas)
    showlinercurve(result_theta, training_data_include_y)


if __name__ == '__main__':
    main()
