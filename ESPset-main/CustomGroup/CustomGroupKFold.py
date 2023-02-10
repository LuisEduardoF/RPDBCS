from sklearn.model_selection._split import GroupKFold
from skorch.helper import SliceDict
import numpy as np


def generator(list):
    for i in range(len(list)):
        yield list[i]


class CustomGroupKFold(GroupKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits)

    def split(self, X, y=None, groups=None):
        list = []
        for iteration in super().split(X, y, groups):
            train, test = iteration    
            if (isinstance(X, SliceDict)):
                X['test'][train] = np.zeros(train.shape)
                X['test'][test] = np.ones(test.shape)
                train = np.append(train, test)
            list.append((train, test))
        return generator(list)
