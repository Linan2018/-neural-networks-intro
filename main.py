# coding=utf-8
from nn import NeuralNetwork, softmax
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# %matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.figsize'] = (8, 6)


def task1():
    # 二分类
    net = NeuralNetwork([2, 4, 1], activation='line', softmax_=False)

    train_N = 200
    test_N = 100

    x = np.random.normal(loc=0.0, scale=2.0, size=(train_N, 2))

    a = 1.0
    b = 0.15
    f = lambda x: a * x + b

    plt.figure(1)
    plt.plot(x, f(x), 'g', label='真实分割线')

    # 线性分割前面的点
    y = np.zeros([train_N, 1])

    for i in range(train_N):
        if f(x[i, 0]) >= x[i, 1]:
            # 点在直线下方
            y[i] = 1
            plt.plot(x[i, 0], x[i, 1], 'bo', markersize=8, label='类一')
        else:
            # 点在直线上方
            y[i] = -1
            plt.plot(x[i, 0], x[i, 1], 'ro', markersize=8, label='类二')

    plt.legend(labels=['真实分割线'], loc=1)
    plt.title('随机数生成及展示')
    plt.show()

    wb = net.train(x, y, epochs=100, lr=0.001, batchsize=8)

    newx = np.random.normal(loc=0.0, scale=2.0, size=(test_N, 2))
    y_preds = np.array(list(map(net.forward, newx, (wb for _ in range(len(newx))))))

    plt.figure(2)
    plt.plot(x, f(x), 'g', label='真实分割线')
    for i in range(test_N):
        if y_preds[i][0] > 0:
            plt.plot(newx[i, 0], newx[i, 1], 'b^', markersize=8, label='类一（预测）')
        else:
            plt.plot(newx[i, 0], newx[i, 1], 'r^', markersize=8, label='类二（预测）')

    plt.legend(labels=['真实分割线'], loc=1)
    # plt.plot(x, f(x), 'y')
    # plt.legend()
    plt.show()


def task2():
    # 回归
    x_data = np.linspace(-4, 4, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.1, x_data.shape)
    f = lambda x: np.sin(x)
    # y_data = np.square(x_data) + noise
    y_data = f(x_data) + noise

    plt.scatter(x_data, y_data)
    plt.show()

    net = NeuralNetwork([1, 4,  1], activation='tanh', softmax_=False)

    wb = net.train(x_data, y_data, epochs=500, lr=0.005, batchsize=8)
    newx = np.linspace(-4, 4, 50)
    y_preds = np.array(list(map(net.forward, newx, (wb for _ in range(len(newx))))))

    plt.scatter(x_data, y_data)
    plt.plot(newx, y_preds[:, 0, 0], 'r-', lw=2)
    plt.show()

def task3():
    train_N = 100
    test_N = 100

    x1 = np.random.normal(loc=0.0, scale=4.0, size=(train_N, 2)) + [-10, 10]
    x2 = np.random.normal(loc=0.0, scale=4.0, size=(train_N, 2)) + [10, 10]
    x3 = np.random.normal(loc=0.0, scale=4.0, size=(train_N, 2)) + [-10, -10]
    x4 = np.random.normal(loc=0.0, scale=4.0, size=(train_N, 2)) + [10, -10]

    y1 = np.array([[1., 0., 0., 0.] for _ in range(train_N)])
    y2 = np.array([[0., 1., 0., 0.] for _ in range(train_N)])
    y3 = np.array([[0., 0., 1., 0.] for _ in range(train_N)])
    y4 = np.array([[0., 0., 0., 1.] for _ in range(train_N)])

    plt.plot(x1[:, 0], x1[:, 1], 'ro')
    plt.plot(x2[:, 0], x2[:, 1], 'yo')
    plt.plot(x3[:, 0], x3[:, 1], 'bo')
    plt.plot(x4[:, 0], x4[:, 1], 'go')
    plt.show()

    x = np.vstack((x1, x2, x3, x4))
    y = np.vstack((y1, y2, y3, y4))

    net = NeuralNetwork([2, 4, 4], activation='relu', softmax_=True)

    wb = net.train(x, y, loss='cross_entropy', epochs=200, lr=0.01, batchsize=2)
    # print("over")
    newx1 = np.random.normal(loc=0.0, scale=4.0, size=(test_N, 2)) + [-10, 10]
    newx2 = np.random.normal(loc=0.0, scale=4.0, size=(test_N, 2)) + [10, 10]
    newx3 = np.random.normal(loc=0.0, scale=4.0, size=(test_N, 2)) + [-10, -10]
    newx4 = np.random.normal(loc=0.0, scale=4.0, size=(test_N, 2)) + [10, -10]

    newx = np.vstack((newx1, newx2, newx3, newx4))

    y_preds = np.array(list(map(net.forward, newx, (wb for _ in range(len(newx))))))
    # print(y_preds.shape)
    # y_preds = np.array([softmax(a) for a in np.squeeze(y_preds)])
    print(y_preds)
    # print(y_preds)

    sty = ['r^', 'y^', 'b^', 'g^']

    plt.figure(2)

    for i in range(test_N):
        plt.plot(newx[i, 0], newx[i, 1], sty[int(np.argmax(y_preds[i]).max())], markersize=8, label='类一（预测）')
    plt.show()


def task4():
    # soft二分类
    net = NeuralNetwork([2, 4, 2],  activation='tanh', softmax_=True)

    train_N = 200
    test_N = 100

    x = np.random.normal(loc=0.0, scale=2.0, size=(train_N, 2))

    a = 1.0
    b = 0.15
    f = lambda x: a * x + b

    plt.figure(1)
    plt.plot(x, f(x), 'g', label='真实分割线')

    # 线性分割前面的点
    y = np.zeros([train_N, 2])

    for i in range(train_N):
        if f(x[i, 0]) >= x[i, 1]:
            # 点在直线下方
            y[i] = np.array([1., 0.])
            plt.plot(x[i, 0], x[i, 1], 'bo', markersize=8, label='类一')
        else:
            # 点在直线上方
            y[i] = np.array([0., 1.])
            plt.plot(x[i, 0], x[i, 1], 'ro', markersize=8, label='类二')

    plt.legend(labels=['真实分割线'], loc=1)
    plt.title('随机数生成及展示')
    plt.show()
    wb = net.train(x, y, loss='cross_entropy', epochs=100, lr=0.001, batchsize=8)

    # wb = net.train(x, y, softmax_=True, loss='cross_entropy', epochs=200, lr=0.001, batchsize=8)

    newx = np.random.normal(loc=0.0, scale=2.0, size=(test_N, 2))
    y_preds = np.array(list(map(net.forward, newx, (wb for _ in range(len(newx))))))
    # y_preds = softmax(np.squeeze(y_preds))
    y_preds = np.array([softmax(a) for a in np.squeeze(y_preds)])

    plt.figure(2)
    plt.plot(x, f(x), 'g', label='真实分割线')
    # print(y_preds.shape)
    for i in range(test_N):
        if y_preds[i][0][0] > 0.5:
            plt.plot(newx[i, 0], newx[i, 1], 'b^', markersize=8, label='类一（预测）')
        else:
            plt.plot(newx[i, 0], newx[i, 1], 'r^', markersize=8, label='类二（预测）')

    plt.legend(labels=['真实分割线'], loc=1)
    # plt.plot(x, f(x), 'y')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    task4()
