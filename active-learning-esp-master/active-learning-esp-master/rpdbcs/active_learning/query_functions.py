from sklearn.base import BaseEstimator
import numpy as np
from modAL.utils.data import modALinput
from modAL import ActiveLearner
from scipy.spatial import distance_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def queryfunction_wrapper(query_function):
    """
    Some queries functions need X0 (the initial/current train data) and others not.
    This function makes those that don't need, have the same interface/parameters.
    """
    return lambda classifier, X, X0, n_instances, **kwargs: query_function(classifier, X, n_instances, **kwargs)


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
