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
            for sample in rd.permutation(data):
                self.optimisation(sample)

    def optimisation(self, sample):
        y = int(sample[0])
        x = [1]
        x.extend(sample[1:])
        diff = [self.eta * (((1 / (1 + self.__exp_wx(x))) - y) * x_i) for x_i in x]
        self.w = [w_i - diff_i for w_i, diff_i in zip(self.w, diff)]

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

    def predict_prob(self, data):
        pred = []
        for d in data:
            x = [1]
            x.extend(d[1:])
            pred.append(1 / (1 + self.__exp_wx(x)))
        return np.array(pred)

    def predict(self, data, threshold=0.5):
        pred_prob = self.predict_prob(data)
        pred = np.array([1 if p >= threshold else 0 for p in pred_prob])
        return pred


def sampling(num=50, min=0, range=5):
    return np.array(rd.rand(num) * range + min)

if __name__ == '__main__':
    pos_sample = np.array([[1, x, y] for x, y in zip(sampling(), sampling(min=5))])
    neg_sample = np.array([[0, x, y] for x, y in zip(sampling(min=5), sampling())])
    sample = np.r_[pos_sample, neg_sample]
    rd.shuffle(sample)
    lr = LogisticRegression(eta=0.01)
    lr.train(sample)
    lr.show_params()
    pred = lr.predict(sample)
    label = np.array([int(d[0]) for d in sample])
    print(confusion_matrix(label, pred))
