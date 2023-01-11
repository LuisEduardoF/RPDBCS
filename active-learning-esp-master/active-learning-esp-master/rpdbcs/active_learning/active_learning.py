from sklearn.model_selection._split import _BaseKFold
from modAL.models import ActiveLearner
import numpy as np


# {'fit_time': array([0.97448158]), 'score_time': array([0.18386269]),
# 'test_accuracy': array([0.93539764]), 'test_f1_macro': array([0.82236397]), 'test_precision_macro': array([0.82481409]), 'test_recall_macro': array([0.82047795]), 'test_log_loss': array([-2.23128651]), 'test_fold count': array([4489]), 'test_f-measure_Roçamento': array([0.66972477]), 'test_precision_Roçamento': array([0.65765766]), 'test_recall_Roçamento': array([0.68224299]), 'test_f-measure_Problemas na medição': array([0.89473684]), 'test_precision_Problemas na medição': array([0.92727273]), 'test_recall_Problemas na medição': array([0.86440678]), 'test_f-measure_Desalinhamento': array([0.73809524]), 'test_precision_Desalinhamento': array([0.73809524]), 'test_recall_Desalinhamento': array([0.73809524]), 'test_f-measure_Desbalanceamento': array([0.84236453]), 'test_precision_Desbalanceamento': array([0.83414634]), 'test_recall_Desbalanceamento': array([0.85074627]), 'test_f-measure_Normal': array([0.96689847]), 'test_precision_Normal': array([0.96689847]), 'test_recall_Normal': array([0.96689847])}

def active_learning(active_estimator: ActiveLearner, Xpool, Ypool, Xtest, Ytest, query_size: int, query_budget: int, metrics):
    """
    Peforms active learning with speficied estimator and dataset.
    Args:
        metrics (dict): str->callback function.

    Returns:
        A dictionary where the key is the name of a metric (from parameter `metrics`) or statistic about the experiment.
        The value of the dict is a numpy array of size ceil(query_budget/query_size) + 1`
    """
    preds = active_estimator.predict(Xtest)
    Result = {'test_'+name: [m(Ytest, preds)] for name, m in metrics.items()}
    queried_samples = [0]
    queried_idxs = []
    while(query_budget > 0):
        if(query_budget < query_size):
            query_size = query_budget
        query_idx, _ = active_estimator.query(Xpool, X0=active_estimator.X_training,
                                              n_instances=query_size)  # TODO: see active_estimator.Xt_training
        active_estimator.teach(Xpool[query_idx], Ypool[query_idx])
        Xpool = np.delete(Xpool, query_idx, axis=0)
        Ypool = np.delete(Ypool, query_idx, axis=0)
        preds = active_estimator.predict(Xtest)
        for name, m in metrics.items():
            Result['test_'+name].append(m(Ytest, preds))
        queried_samples.append(queried_samples[-1]+query_size)
        queried_idxs.append(np.array(query_idx))
        query_budget -= query_size
    Result['queried samples'] = queried_samples
    Result['queried idxs'] = queried_idxs
    return {name: np.array(r) for name, r in Result.items()}
