#! /usr/bin/env python

import numpy as np


def _grad(y_true, y_pred, alpha, gamma, epsilon):
    grad = 0
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            grad += gamma * (yp ** (gamma - 1)) * np.log(1 - yp) - (yp ** gamma) / (1 - yp)
        else:
            grad += -gamma * ((1 - yp) ** (gamma - 1)) * np.log(yp) - ((1 - yp) ** gamma) / yp
    return -alpha * grad


def _hess(y_true, y_pred, alpha, gamma, epsilon):
    hess = 0
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            hess += gamma * (gamma - 1) * (yp ** (gamma - 2)) * np.log(1 - yp) \
                    - 2 * gamma * (yp ** (gamma - 1)) / (1 - yp) \
                    + (yp ** gamma) / ((1 - yp) ** 2)
        else:
            hess += gamma * (gamma - 1) * ((1 - yp) ** (gamma - 2)) * np.log(yp) \
                    - 2 * gamma * ((1 - yp) ** (gamma - 1)) / yp \
                    - ((1 - yp) ** gamma) / (yp ** 2)
    return -alpha * hess


def gen_focalobj(alpha=0.25, gamma=2.0, epsilon=1e-15):
    def focalobj(preds, dtrain):
        labels = dtrain.get_label()
        grad = _grad(labels, preds, alpha, gamma, epsilon)
        hess = _hess(labels, preds, alpha, gamma, epsilon)
        return grad, hess
    return focalobj
