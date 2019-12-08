import numpy as np
import copy
import random
import matplotlib.pyplot as plt

# from functools import reduce

# %matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.figsize'] = (8, 6)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def cross_entropy_loss(y_true, y_pred):
    return (-1) * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum(axis=1).mean()


def line(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    # 防止溢出
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    temp = x - np.max(x, axis=1).reshape((x.shape[0], 1))
    exp_x = np.exp(temp)
    return exp_x / np.sum(exp_x, axis=1).reshape((x.shape[0], 1))


def xavier_init(fan_in, fan_out, size, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low, high, size=size)


class NeuralNetwork:
    def __init__(self, layers, activation, softmax_):

        self.activation = activation
        self.softmax_ = softmax_

        fan_in = layers[0]
        fan_out = layers[-1]

        n_hlayers = len(layers)

        w = [xavier_init(fan_in, fan_out, size=(layers[i], layers[i + 1])) for i in range(n_hlayers - 1)]
        b = [xavier_init(fan_in, fan_out, size=(1, 1)) for _ in range(n_hlayers - 1)]

        self.wb = [w, b]

    def train(self, data, all_y_true, batchsize=4, loss='mse', epochs=10, lr=0.00001):
        n = len(data)

        for epoch in range(1, epochs + 1):
            dataset = list(zip(data, all_y_true))
            random.shuffle(dataset)
            # print(dataset)

            n_batch = np.ceil(n / batchsize)
            for index in range(int(n_batch)):
                l_bound = index * batchsize
                r_bound = (index + 1) * batchsize

                if r_bound > n:
                    r_bound = n
                    l_bound = r_bound - batchsize

                x_batch = np.zeros((r_bound - l_bound, *data.shape[1:]))
                y_batch = np.zeros((r_bound - l_bound, *all_y_true.shape[1:]))

                instance_count = 0
                for x, y in dataset[l_bound:r_bound]:
                    x_batch[instance_count] = x
                    y_batch[instance_count] = y
                    instance_count += 1
                self.sgd(x_batch, y_batch, self.wb, lr=lr, loss=loss)

            if epoch % 10 == 0:
                y_preds = np.array(list(map(self.forward, data, (self.wb for _ in range(len(data))))))[:, :, 0]
                lossVal = mse_loss(all_y_true, y_preds)
                print("Epoch %d \t loss: %.6f " % (epoch, lossVal))
        return self.wb

    def forward(self, x, wb):
        """
        前向传播
        """

        if self.activation == 'relu':
            actv = relu
        elif self.activation == 'tanh':
            actv = tanh
        elif self.activation == 'sigmoid':
            actv = sigmoid
        elif self.activation == 'line':
            actv = line
        else:
            actv = None

        result = np.copy(x)
        for w, b in zip(wb[0], wb[1]):
            result = actv(np.dot(result, w) + b)
        if self.softmax_:
            result = softmax(result)

        return result

    def grad(self, x, wb, i, y_true, loss='mse', d=1e-6):
        """
        求偏导
        """
        loss_func = mse_loss
        if loss == 'mse':
            loss_func = mse_loss
        elif loss == 'cross_entropy':
            loss_func = cross_entropy_loss
        wb_ = copy.deepcopy(wb)
        o = self.forward(x, wb_)
        lossValue = loss_func(np.squeeze(y_true), np.squeeze(o))

        wb_[i[0]][i[1]][i[2:]] += d
        o_ = self.forward(x, wb_)
        lossValue_ = loss_func(np.squeeze(y_true), np.squeeze(o_))
        return (lossValue_ - lossValue) / d

    def sgd(self, x, y_true, wb, lr=0.1, loss='mse'):
        """
        梯度下降
        """
        _, i, j, k = 0, 0, 0, 0
        for _ in range(2):
            for k in range(len(wb[_])):
                for i in range(wb[_][k].shape[0]):
                    for j in range(wb[_][k].shape[1]):
                        wb[_][k][i, j] -= lr * self.grad(x, wb, (_, k, i, j), y_true, loss=loss)
