from functools import reduce
from sklearn.base import BaseEstimator
import numpy as np
from modAL.utils.data import modALinput
from modAL import ActiveLearner
from modAL.uncertainty import classifier_entropy
from scipy.spatial import distance_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class sequential_query:
    def __init__(self, *qfuncs):
        self.qfuncs = qfuncs
        self.multipliers = list(range(len(qfuncs), 0, -1))

    def __call__(self, classifier, X, X0, n_instances, **kwargs) -> np.ndarray:
        X_ = X
        idx_list = []
        for qfunc, mm in zip(self.qfuncs, self.multipliers):
            m_instances = mm*n_instances
            if m_instances >= X_.shape[0]:
                print("m_instances >= X_.shape[0]")
                # For some AssertionErrors
                idx = np.arange(X_.shape[0])
            else:
                idx = qfunc(classifier=classifier, X=X_, X0=X0,
                            n_instances=m_instances, **kwargs)
                if isinstance(idx, list):
                    # List of ints does not work on reduce statement
                    # Investigate later
                    idx = np.array(idx)
                if isinstance(idx, tuple):
                    idx, *_ = idx
            idx_list.append(idx)
            X_ = X_[idx]
        Xf = reduce(lambda x, y: y[x], idx_list[::-1])
        return Xf

    def __repr__(self):
        return f'sequential_query({self.qfuncs})'

    def __str__(self):
        return repr(self)


class NewFaultQuery:
    def __init__(self, new_fault_idx: int):
        """
        `new_fault_idx`: Number that represents the new fault
        """
        self.fault_index = new_fault_idx

    def __call__(self, classifier, X, n_instances, **predict_proba_kwargs):
        clf = classifier
        x_proba = clf.predict_proba(X)
        assert x_proba.ndim == 2, "x_proba com mais de duas dimensões"

        x_proba_max = x_proba.max(axis=1)
        x_proba_new = x_proba[:, self.fault_index]
        margin = x_proba_max - x_proba_new
        entropy = classifier_entropy(clf, X)
        assert margin.shape == entropy.shape, "margin e entropy com formatos diferentes"
        assert margin.ndim == 1
        assert entropy.ndim == 1

        query_indexes = np.lexsort([-entropy, margin])[:n_instances]
        return query_indexes, X[query_indexes]


class NewFaultWithMargin:
    """
    New Fault margin + "1-2" margin
    """

    def __init__(self, new_fault_idx: int):
        """
        `new_fault_idx`: Number that represents the new fault
        """
        self.fault_index = new_fault_idx

    def __call__(self, classifier, X, n_instances, **predict_proba_kwargs):
        clf = classifier
        x_proba = clf.predict_proba(X)
        assert x_proba.ndim == 2, "x_proba com mais de duas dimensões"

        x_proba_max = x_proba.max(axis=1)
        x_proba_new = x_proba[:, self.fault_index]
        new_fault_margin = x_proba_max - x_proba_new

        x_proba_psort = np.partition(-x_proba, 1, 1)
        x_1max, x_2max = -x_proba_psort[:, 0], -x_proba_psort[:, 1]
        margin_1_2 = x_1max - x_2max

        assert new_fault_margin.shape == margin_1_2.shape, "new fault e 1-2 com formatos diferentes"
        assert new_fault_margin.ndim == 1

        combined = new_fault_margin + margin_1_2
        query_indexes = np.lexsort([combined])[:n_instances]
        return query_indexes, X[query_indexes]


def queryfunction_wrapper(query_function):
    """
    Some queries functions need X0 (the initial/current train data) and others not.
    This function makes those that don't need, have the same interface/parameters.
    """
    return lambda classifier, X, X0, n_instances, **kwargs: query_function(classifier, X, n_instances, **kwargs)


def queryrandom_wrapper(query_function):
    """
    Like `queryfunction_wrapper` but with random sampling before
    """
    def random_query(classifier, X, X0, n_instances, **kwargs):
        nonlocal query_function

        idx = random_sampling(classifier, X, None, n_instances=2*n_instances, **kwargs)
        idx2 = query_function(classifier, X[idx], n_instances, **kwargs)  # needs a better name
        return idx[idx2]

    return random_query


def random_sampling(classifier: BaseEstimator, X: modALinput, X0, n_instances: int = 1, **kwargs) -> np.ndarray:
    """
    Gets n_instances samples from X, randomly chosen.
    """
    return np.random.permutation(range(len(X)))[:n_instances]


def k_center_greedy(classifier: ActiveLearner, X: modALinput, X0, n_instances: int = 1, **kwargs) -> np.ndarray:
    """
    See `Active Learning for Convolutional Neural Networks: A Core-Set Approach <https://arxiv.org/abs/1708.00489>`_.
    """
    def transform_data(estimator, X):
        if(isinstance(estimator, GridSearchCV)):
            return transform_data(estimator.best_estimator_, X)
        if(isinstance(estimator, Pipeline)):
            for _, _, transform in estimator._iter(with_final=False):
                X = transform.transform(X)
            return X
        return estimator.transform(X)

    n = len(X)
    m = len(X0)
    estimator = classifier.estimator
    X = transform_data(estimator, X)
    X0 = transform_data(estimator, X0)
    D = distance_matrix(X, X0)
    Dmin = D.min(axis=1)
    ret = []
    for i in range(n_instances):
        k = Dmin.argmax()
        ret.append(k)
        Dmin = np.minimum(Dmin, distance_matrix(X, X[k:k+1]).min(axis=1))
        Dmin[i] = -1

    '''
    import seaborn as sns
    import pandas as pd
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    XX = np.vstack([X,X0])
    XX = PCA(2).fit_transform(XX)

    df = pd.DataFrame(XX, columns=['pca1','pca2'])
    df['selection']=['not_selected']*n+['initial']*m
    df['selection'][ret] = 'selected'
    df.to_csv('/tmp/tmp_dataframe.csv', index=False)
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='selection')
    plt.show()
    '''
    return ret


def bald(classifier: ActiveLearner, X: modALinput, n_instances: int = 1, **kwargs):
    """
    See `Bayesian Active Learning for Classification and Preference Learning <https://arxiv.org/abs/1112.5745>`_.
    """
    estimator = classifier.estimator
    preds = np.array([estimator.predict_proba_stochastic(X) for i in range(20)])
    preds[preds < 1e-6] = 1e-6 # Because of entropy: p*log(p)

    Pg = preds.mean(axis=0)
    Eg = -Pg*np.log2(Pg)
    Eg = Eg.sum(axis=1)

    Ew = -preds*np.log2(preds)
    Ew = Ew.sum(axis=2)
    Ew = Ew.mean(axis=0)

    I = Eg - Ew
    return np.argsort(I)[-n_instances:]
