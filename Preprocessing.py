import numpy as np


def mean_normalize(X, axis=0):
    return (X - np.mean(X, axis=axis, keepdims=True)) / np.std(X, axis=axis, keepdims=True)


def normalize(X, axis=0):
    return (X - np.min(X, axis=axis, keepdims=True)) / (
                np.max(X, axis=axis, keepdims=True) - np.min(X, axis=axis, keepdims=True))


def to_one_hot(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T
