#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from numpy import random as rd
from sklearn.metrics import confusion_matrix


class LogisticRegression():

    def __init__(self, dim=2, eta=0.1, iter=100):
        self.dim = dim
        self.eta = eta
        self.iter = iter
        self.w = np.array([self.__init_param() for i in range(0, self.dim + 1)])

    def __init_param(self):
        return rd.rand() * 2 + -1

    def train(self, data):
        for i in range(self.iter):
            self.optimisation(data)

    def optimisation(self, data):
        diff = [0 for i in range(0, self.dim + 1)]
        for d in data:
            y = int(d[0])
            x = [1]
            x.extend(d[1:])
            diff = [diff_i + (((1 / (1 + self.__exp_wx(x))) - y) * x_i) for diff_i, x_i in zip(diff, x)]
        self.w = [w_i - diff_i for w_i, diff_i in zip(self.w, [self.eta * p for p in diff])]

    def __exp_wx(self, x):
        return np.exp(-1 * self.__wx(x))

    def __wx(self, x):
        wx = 0
        for i in range(0, self.dim + 1):
            wx += self.w[i] * x[i]
        return wx

    def show_params(self):
        for i, w in enumerate(self.w):
            print("w[{}]: {}".format(i, w))

    def predict(self, data):
        pred = []
        for d in data:
            x = [1]
            x.extend(d)
            pred.append(1 / (1 + self.__exp_wx(x)))
        return np.array(pred)


def sampling(num=50, min=0, range=5):
    return np.array(rd.rand(num) * range + min)

if __name__ == '__main__':
    pos_sample = np.array([[1, x, y] for x, y in zip(sampling(), sampling(min=5))])
    neg_sample = np.array([[0, x, y] for x, y in zip(sampling(min=5), sampling())])
    sample = np.r_[pos_sample, neg_sample]
    rd.shuffle(sample)
    lr = LogisticRegression(iter=1000, eta=0.01)
    lr.train(sample)
    lr.show_params()
    pred = lr.predict(sample)
    label = [int(d[0]) for d in sample]
    pred_label = [1 if p >= 0.5 else 0 for p in pred]
    print(confusion_matrix(label, pred_label))
