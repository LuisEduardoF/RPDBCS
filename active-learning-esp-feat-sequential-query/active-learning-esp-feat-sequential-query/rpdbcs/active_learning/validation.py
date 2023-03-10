from active_learning import active_learning
import numpy as np
import random
import pandas as pd
import torch
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames
import skorch
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
import query_functions
from query_functions import queryfunction_wrapper
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, PredefinedSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from tripletnet.networks import TripletNetwork, lmelloEmbeddingNet
from tripletnet.datahandler import BalancedDataLoader
from tripletnet.callbacks import LoadEndState, LRMonitor, CleanNetCallback
from tripletnet.TripletNetClassifierMCDropout import TripletNetClassifierMCDropout
import itertools
from tempfile import mkdtemp
from shutil import rmtree
from adabelief_pytorch import AdaBelief
from rpdbcs_yaml import load_yaml
from tripletnet.utils import PipelineExtended
import sklearn
from pathlib import Path
from splitter import KindaStratifiedShuffleSplit, ConfigTrainSplitter
import sys
import joblib

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEEP_CACHE_DIR = mkdtemp()
PIPELINE_CACHE_DIR = mkdtemp()

CONFIG = None


def loadRPDBCSData(data_dir, nsigs=100000):
    """
    Signal are normalized with respect to 37.2894Hz.
    Multi-label samples are discarded here.
    """
    D = readDataset(data_dir / 'freq.csv', data_dir / 'labels.csv',
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32,
                    discard_multilabel=False)
    D.discardMultilabel()
    targets, _ = D.getMulticlassTargets()
    df = D.asDataFrame()
    # D.remove(np.where(targets == 3)[0])  # removes desalinhamento
    # df = D.asDataFrame()
    # D.remove(df[(df['project name'] == 'Baker') & (df['bcs name'] == 'MA15')].index.values)
    print("Dataset length", len(D))
    D.normalize(37.28941975)
    # D.shuffle()

    return D


def getBaseClassifiers(pre_pipeline=None, config=None):
    """
    Gets all traditional machine learning classifier that will be use in the experiments.
    They will be used in both TripletNet space and the Hand-crafted space.
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    clfs = []
    knn = KNeighborsClassifier()
    knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
    rf = RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1)
    rf_param_grid = {'max_features': [2, 3, 4, 5]}
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE, min_impurity_decrease=0.001)
    qda = QuadraticDiscriminantAnalysis()
    qda_param_grid = {'reg_param': [0.0, 1e-6, 1e-5]}

    clfd = {
        "knn": (knn, knn_param_grid),
        "DT": (dtree, {}),
        "RF": (rf, rf_param_grid),
        "NB": (GaussianNB(), {}),
        "QDA": (qda, qda_param_grid),
    }

    # clfs.append(("knn", knn, knn_param_grid))
    # clfs.append(("DT", dtree, {}))
    # clfs.append(("RF", rf, rf_param_grid))
    # clfs.append(("NB", GaussianNB(), {}))
    # clfs.append(("QDA", qda, qda_param_grid))

    if config is None:
        clfs = [(i, *j) for i, j in clfd.items()]
    else:
        clfs = [(i, *j) for i, j in clfd.items() if i in config.base_clf]

    if(pre_pipeline is not None):
        return [(cname, Pipeline([pre_pipeline, ('base_clf', c)]), {"base_clf__%s" % k: v for k, v in pgrid.items()})
                for cname, c, pgrid in clfs]

    return clfs


def getCallbacks():
    """
    Callbacks used by the neural network.
    One of the callbacks is monitoring and saving the best epoch (lowest non zero triplets), 
        so that at the end of training the best is loaded and actually used for predictions.
    """
    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR,
                                                      monitor='non_zero_triplets_best')  # monitor='train_loss_best')
    # lrscheduler = skorch.callbacks.LRScheduler(policy=optim.lr_scheduler.StepLR, step_size=50, gamma=0.8,
    #                                            event_name=None)
    # ?? poss??vel dar nomes ao callbacks para poder usar gridsearch neles: https://skorch.readthedocs.io/en/stable/user/callbacks.html#learning-rate-schedulers

    callbacks = [('non_zero_triplets', skorch.callbacks.PassthroughScoring(name='non_zero_triplets', on_train=True))]
    callbacks += [checkpoint_callback, LoadEndState(checkpoint_callback)]  # , LRMonitor()]
    callbacks += [skorch.callbacks.EarlyStopping(monitor="train_loss", patience=100, threshold=1e-4)]
    return callbacks


def createNeuralClassifier():
    """
    Common neural net classifier.
    """
    from siamese_triplet.networks import ClassificationNet

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR,
                                                      monitor='train_loss_best')
    lrscheduler = skorch.callbacks.LRScheduler(policy=optim.lr_scheduler.StepLR, step_size=50, gamma=0.8,
                                               event_name=None)
    # ?? poss??vel dar nomes ao callbacks para poder usar gridsearch neles: https://skorch.readthedocs.io/en/stable/user/callbacks.html#learning-rate-schedulers

    callbacks = [checkpoint_callback, LoadEndState(checkpoint_callback)]

    parameters = {
        'callbacks': callbacks,
        'device': 'cuda',
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'module__embedding_net': lmelloEmbeddingNet(8), 'module__n_classes': 5}
    parameters = {**parameters, **optimizer_parameters}
    convnet = skorch.NeuralNetClassifier(ClassificationNet, **parameters)
    return ('convnet', convnet, {})


def getDeepTransformers():
    """
    Constructs and returns Triplet Networks.
    """
    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    parameters = {
        'callbacks': getCallbacks(),
        'device': 'cuda',
        'module': lmelloEmbeddingNet,
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        # 'criterion__triplet_selector': siamese_triplet.utils.HardestNegativeTripletSelector(1.0),
        'margin_decay_value': 0.75, 'margin_decay_delay': 100}
    parameters = {**parameters, **optimizer_parameters}
    deep_transf = []
    tripletnet = TripletNetwork(module__num_outputs=8, init_random_state=100, **parameters)
    tripletnetMCDroupout = TripletNetClassifierMCDropout(None, mc_iters=20, module__num_outputs=8,
                                                         cache_dir=PIPELINE_CACHE_DIR, init_random_state=100, **parameters)

    tripletnet_param_grid = {'batch_size': [80],
                             'margin_decay_delay': [500],
                             'module__num_outputs': [8]}
    # deep_transf.append(("tripletnet_mcdropout", tripletnetMCDroupout, tripletnet_param_grid))
    deep_transf.append(("tripletnet", tripletnet, tripletnet_param_grid))

    return deep_transf


def getMetrics(labels_names=None):
    """
    args:
        labels_names (dict): mapping from label code (int) to label name (str).
    returns:
        A dictionary where key is the name of the metric and its value is a callback function receiving two parameters.
    """

    def f1_problem(n):
        assert 0 <= n <= 4

        def f1(y_true, y_pred):
            nonlocal n
            y_trueb = y_true.copy()
            y_trueb[y_true == n] = 1
            y_trueb[y_true != n] = 0

            y_predb = y_pred.copy()
            y_predb[y_pred == n] = 1
            y_predb[y_pred != n] = 0

            return f1_score(y_trueb, y_predb, average='binary')
        return f1

    scoring = {
        'accuracy': accuracy_score,
        'f1_macro': lambda p1, p2: f1_score(p1, p2, average='macro'),
        'precision': lambda p1, p2: precision_score(p1, p2, average='macro'),
        'recall': lambda p1, p2: recall_score(p1, p2, average='macro'),
        'f1_normal': f1_problem(0),
        'f1_rocamento': f1_problem(1),
        'f1_medicao': f1_problem(2),
        'f1_desali': f1_problem(3),
        'f1_desbal': f1_problem(4),
    }
    # if(labels_names is not None):
    #     for code, name in labels_names.items():
    #         scoring['f-measure_%s' % name] = make_scorer(f1_score, average=None, labels=[code])

    return scoring


def combineTransformerClassifier(transformers, base_classifiers):
    """
    Combines the TripletNetwork with a base classifier (ex: K-NN) to form a scikit-learn Pipeline.

    returns:
        A scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """
    from sklearn.ensemble import VotingClassifier
    gridsearch_sampler = KindaStratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=RANDOM_STATE)

    def buildGridSearch(T, base_classif, transf_param_grid, base_classif_param_grid):
        base_classif = GridSearchCV(base_classif, base_classif_param_grid, cv=gridsearch_sampler, n_jobs=-1)
        if(isinstance(T, TripletNetClassifierMCDropout)):
            T.setBaseClassifier(base_classif)
            return T
        clf = PipelineExtended([('transformer', T),
                                ('base_classifier', base_classif)],
                               memory=PIPELINE_CACHE_DIR)
        # transf_param_grid = {"transformer__%s" % k: v for k, v in transf_param_grid.items()}
        # base_classif_param_grid = {"base_classifier__%s" % k: v for k, v in base_classif_param_grid.items()}
        # param_grid = {**transf_param_grid, **base_classif_param_grid}
        # return GridSearchCV_norefit(clf, param_grid, scoring='f1_macro')
        return clf

    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        if(isinstance(transf, list)):
            C = [("net%d" % i, buildGridSearch(T, base_classif, transf_param_grid, base_classif_param_grid))
                 for i, T in enumerate(transf)]
            if('voting' in transf_name):
                classifier = VotingClassifier(estimators=C, voting='soft')
            elif('bagging' in transf_name):
                classifier = TorchBaggingClassifier(base_estimator=C[0][1], n_estimators=len(C), bootstrap=True,
                                                    bootstrap_features=False, random_state=RANDOM_STATE)
        else:
            classifier = buildGridSearch(transf, base_classif, transf_param_grid, base_classif_param_grid)

        yield ('%s + %s' % (transf_name, base_classif_name), classifier)


def SplitActiveLearning(X, Y, init_train_size, test_size=0.25):
    """
    Splits dataset into 3 datasets: initial train dataset, pool dataset and test dataset.
    Pool dataset and test dataset are sampled in a stratified way (same distribution of labels among both datasets).
    Args:
        init_train_size (int or float):  If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split. If int, represents the
            absolute number of train samples.

        test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion
            of the pool dataset to include in the test split. If int, represents the
            absolute number of test samples.

    """
    sampler0 = StratifiedShuffleSplit(n_splits=1, train_size=init_train_size, random_state=RANDOM_STATE)
    idxs_ini, idxs_others = next(sampler0.split(X, Y))
    X0, Y0 = X[idxs_ini], Y[idxs_ini]

    # Here we ensures traindataset have at least 5 samples of each class
    # for i in range(max(Y)):
    #     n = (Y0 == i).sum()
    #     if(n < 5):  # TODO: dont hard-code. Make a parameter for this.
    #         idxs_ini = np.append(idxs_ini, np.where(Y == i)[0][:5-n])
    #         X0, Y0 = X[idxs_ini], Y[idxs_ini]
    Xo, Yo = X[idxs_others], Y[idxs_others]

    sampler_pool = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idxs_pool, idxs_test = next(sampler_pool.split(Xo, Yo))
    Xpool, Ypool = Xo[idxs_pool], Yo[idxs_pool]
    Xtest, Ytest = Xo[idxs_test], Yo[idxs_test]

    return X0, Y0, Xpool, Ypool, Xtest, Ytest


def SplitActiveLearningKFold(X, Y, init_train_size, n_splits=10, hide=None):
    """
    Splits dataset into 3 datasets: initial train dataset, pool dataset and test dataset.
    Pool dataset and test dataset are sampled in a stratified way (same distribution of labels among both datasets).
    Args:
        init_train_size (int or float):  If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split. If int, represents the
            absolute number of train samples.

        test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion
            of the pool dataset to include in the test split. If int, represents the
            absolute number of test samples.

    """
    if isinstance(n_splits, float):
        assert 0 <= n_splits <= 1
        n_splits = round(1/n_splits)
    sampler0 = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    for idxs_others, idxs_test in sampler0.split(X, Y):
        Xtest, Ytest = X[idxs_test], Y[idxs_test]
        Xo, Yo = X[idxs_others], Y[idxs_others]
        # import ipdb; ipdb.set_trace()

        if CONFIG.init_train_config is None:
            # Split pool and init train set
            sampler_pool = StratifiedShuffleSplit(n_splits=1, test_size=init_train_size, random_state=RANDOM_STATE)
            idxs_pool, idxs_ini = next(sampler_pool.split(Xo, Yo))

            # Here we ensures traindataset have at least 5 samples of each class
            for i in range(max(Y)+1):
                n = (Yo[idxs_ini] == i).sum()
                if n < 5:  # TODO: dont hard-code. Make a parameter for this.
                    append_idxs = np.where(Yo[idxs_pool] == i)[0][:5-n]  # Remove from pool and append to init dataset
                    append_idxs = idxs_pool[append_idxs]
                    idxs_ini = np.append(idxs_ini, append_idxs)
                    idxs_pool = np.setdiff1d(idxs_pool, idxs_ini, assume_unique=True)
                    assert (Yo[idxs_ini] == i).sum() >= 5, "woops..."

            if hide is not None:
                assert 0 <= hide <= 4, "'hide' must be greater than 0 and less than 4"
                print(f"Hiding class {hide}", file=sys.stderr)
                hide_idx = np.where(Yo[idxs_ini] == hide)[0]
                ini_hide_idx = idxs_ini[hide_idx]
                new_idxs_ini = np.setdiff1d(idxs_ini, ini_hide_idx[:-1], assume_unique=True)  # remove first n-1 of n samples
                new_idxs_pool = np.append(idxs_pool, ini_hide_idx[:-1])  # append those samples to pool
                idxs_ini = new_idxs_ini
                idxs_pool = new_idxs_pool
            X0, Y0 = Xo[idxs_ini], Yo[idxs_ini]
            Xpool, Ypool = Xo[idxs_pool], Yo[idxs_pool]
        elif isinstance(CONFIG.init_train_config, int):
            nclass = CONFIG.init_train_config
            # import ipdb; ipdb.set_trace()
            assert nclass >= 1 and nclass <= 4, f"init_train_config inv??lido: {nclass}"
            sampler_pool = StratifiedShuffleSplit(n_splits=1, test_size=5/9, random_state=RANDOM_STATE)
            idxs_pool, idxs_ini = next(sampler_pool.split(Xo, Yo))
            mask = Yo[idxs_ini] == nclass
            pos = np.where(mask)[0]
            if len(pos) > 2:
                idxs_ini = np.delete(idxs_ini, pos[2:])
            X0, Y0 = Xo[idxs_ini], Yo[idxs_ini]
            Xpool, Ypool = Xo[idxs_pool], Yo[idxs_pool]
        else:
            # import ipdb; ipdb.set_trace()
            csplitter = ConfigTrainSplitter(CONFIG.init_train_config)
            if hide is not None:
                # Maybe useless
                # Xo, Yo = Xo[Yo != hide], Yo[Yo != hide]
                pass
            idxs_pool, idxs_ini = next(csplitter.split(Xo, Yo))

            if hasattr(CONFIG, 'init_pool_max_config'):
                pool_config = CONFIG.init_pool_max_config
                for label, pool_max in enumerate(pool_config):
                    if pool_max is None:
                        continue
                    idx = np.where(Yo[idxs_pool] == label)[0]
                    if idx.size > pool_max:
                        idxs_pool = np.delete(idxs_pool, idx[pool_max:])
                    break

            Xpool, Ypool = Xo[idxs_pool], Yo[idxs_pool]
            X0, Y0 = Xo[idxs_ini], Yo[idxs_ini]

        yield X0, Y0, Xpool, Ypool, Xtest, Ytest


def iterateActiveLearners(estimator: sklearn.base.BaseEstimator, X0, Y0, query_strategies, estimator_name):
    """
    Transforms a scikit-learn :class:`~sklearn.base.BaseEstimator` into an active learner of :class:`modAL.ActiveLearner`.

    returns:
        Generator where each item is a tuple of (str,:class:`modAL.ActiveLearner`).
    """
    for qstrat_name, qstrat in query_strategies.items():
        new_estimator_name = "%s [%s]" % (estimator_name, qstrat_name)
        print(new_estimator_name)
        if('bayes' in qstrat_name):
            aclearner = BayesianOptimizer(estimator=estimator, X_training=X0, y_training=Y0,
                                          query_strategy=qstrat)
        else:
            aclearner = ActiveLearner(estimator=estimator, X_training=X0, y_training=Y0,
                                      query_strategy=qstrat)
        yield new_estimator_name, aclearner


def run_active_learning_kfold(clf: sklearn.base.BaseEstimator, X, Y, config, scoring, clf_name) -> list:
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    results = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        X0, Y0, Xpool, Ypool, Xval, Yval = SplitActiveLearning(
            X_train,
            y_train,
            init_train_size=config.init_train_size,
            test_size=config.test_size
        )
        r = run_active_learning(clf, X0, Y0, Xpool, Ypool, Xval, Yval,
                                config.query_strategies, config.query_size,
                                config.budget, scoring, clf_name)
        rval = r
        rtest = {}
        for scoring_name, scoring_function in scoring.items():
            score = scoring_function(clf.predict(X_test), y_test)
            rtest[scoring_name] = score
               
        results.append({
            'val': rval,
            'test': rtest,
        })
    return results


def run_active_learning(classifier: sklearn.base.BaseEstimator, X0, Y0,  Xpool, Ypool, Xtest, Ytest,
                        query_strategies, query_size, budget, scoring, classifier_name, hidden=None):
    Results = {}
    cm_list = []
    for estimator_name, aclearner in iterateActiveLearners(classifier, X0, Y0, query_strategies, classifier_name):
        scores = active_learning(aclearner, Xpool, Ypool, Xtest, Ytest, query_size, budget, scoring, hidden, cm_list)
        scores['queried samples'] += len(X0)
        Results[estimator_name] = scores
    return Results, cm_list

def main(config, D, config_path):
    global DEEP_CACHE_DIR, PIPELINE_CACHE_DIR
    import pandas as pd
    save_cm = str(config_path)
    print(save_cm + '.handcraft_cm_lists.pkl')
    print(save_cm + '.triplet_cm_lists.pkl')

    query_strategies = config.query_strategies

    X = np.expand_dims(D.asMatrix()[:, :6100], axis=1)  # Transforms shape (n,10800) to (n,1,6100).
    Y, Ynames = D.getMulticlassTargets()
    print(Ynames)
    # Yset = enumerate(set(Y))
    # Y, Ymap = pd.factorize(Y)
    # Ynames = {i: Ynames[oldi] for i, oldi in enumerate(Ymap)}
    # group_ids = D.groupids('bcs')
    # sampler used on all gridsearches.
    gridsearch_sampler = KindaStratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    scoring = getMetrics(Ynames)

    # (X0,Y0): Initial train dataset.
    X0, Y0, Xpool, Ypool, Xtest, Ytest = SplitActiveLearning(X, Y,
                                                             init_train_size=config.init_train_size,
                                                             test_size=config.test_size)

    # All classifiers scales features to mean=0 and std=1.
    base_classifiers = getBaseClassifiers(('normalizer', StandardScaler()), config=config)

    Results = {}  # All results are stored in this dict. The keys are the name of the classifiers.
    Results_new = {}

    triplet_cm_lists = {}
    if config.train_neuralnet:
        transformers = getDeepTransformers()
        ###TripletNetwork + BaseClassifier Experiments:###
        for classifier_name, classifier in combineTransformerClassifier(transformers, base_classifiers):
            if config.kfolds is not None:
                splitter = SplitActiveLearningKFold(X, Y,
                                                    config.init_train_size,
                                                    config.kfolds,
                                                    config.hide_class)
                for fold, (X0, Y0, Xpool, Ypool, Xtest, Ytest) in enumerate(splitter):
                    f_clf_name = classifier_name + f' fold-{fold}'
                    r, cm_list = run_active_learning(classifier, X0, Y0,  Xpool, Ypool,
                                            Xtest, Ytest, query_strategies,
                                            config.query_size, config.budget,
                                            scoring, f_clf_name, config.hide_class)
                    triplet_cm_lists[f_clf_name] = cm_list
                    Results.update(r)
            else:
                r = run_active_learning(classifier, X0, Y0,  Xpool, Ypool, Xtest, Ytest,
                                        query_strategies, config.query_size, config.budget, scoring, classifier_name)
                # r = run_active_learning_kfold(classifier, X, Y, config, scoring, classifier_name)
                Results.update(r)
        joblib.dump(triplet_cm_lists, save_cm + '.triplet_cm_lists.pkl')

        if config.train_whole_dataset:
            ###Whole train dataset###
            Xf = np.vstack((X0, Xpool))
            Yf = np.hstack((Y0, Ypool))
            for classifier_name, classifier in combineTransformerClassifier(transformers, base_classifiers):
                r = run_active_learning(classifier, Xf, Yf,  None, None, Xtest, Ytest,
                                        {'random': query_strategies['random']}, 1, 0, scoring, classifier_name+" (alldata)")
                Results.update(r)

        ##ConvNet Experiment:###
        # classifier_name, classifier, _ = createNeuralClassifier()
        # for estimator_name, aclearner in iterateActiveLearners(classifier, X0, Y0, query_strategies, classifier_name):
        #     scores = active_learning(aclearner, Xpool, Ypool, Xtest, Ytest, query_size, budget, scoring)
        #     Results[estimator_name] = scores
    handcraft_cm_lists = {}
    if config.train_single_classifiers:
        ###Classifiers applied to ICTAI2016 features###

        # more_query_strats: to avoid spend too much time training one neural network for each query function,
        #   only fast classifier are trained with all query functions.
        more_query_strats = {'top margin': queryfunction_wrapper(uncertainty_sampling),
                             '1-2 margin': queryfunction_wrapper(margin_sampling)}
        # query_strategies = {**query_strategies, **more_query_strats}

        ictaifeats_names = getICTAI2016FeaturesNames()
        features = D.asDataFrame()[ictaifeats_names].values
        X0, Y0, Xpool, Ypool, Xtest, Ytest = SplitActiveLearning(features, Y,
                                                                 init_train_size=config.init_train_size,
                                                                 test_size=config.test_size)
        for classifier_name, classifier, param_grid in base_classifiers:
            # n_jobs: You may not want all your cores being used.
            classifier = GridSearchCV(classifier, param_grid, scoring='f1_macro', n_jobs=-1,
                                      cv=gridsearch_sampler)
            if config.kfolds is not None:
                splitter = SplitActiveLearningKFold(features, Y,
                                                    config.init_train_size,
                                                    config.kfolds,
                                                    config.hide_class)
                for fold, (X0, Y0, Xpool, Ypool, Xtest, Ytest) in enumerate(splitter):
                    f_clf_name = classifier_name + f' fold-{fold}'
                    
                    examples = np.random.choice(np.where(Ytest == 0)[0], 5, replace=False)
                    # examples = random.sample(range(0, len(Xtest)), 5)
                    
                    print("Examples Choosen: {}, Classes: {}".format(examples, Ytest[examples]))
                    X0_new = np.append(X0, Xtest[examples], axis=0)
                    Y0_new = np.append(Y0, Ytest[examples], axis=0)

                    Xtest = np.delete(Xtest, examples, 0)
                    Ytest = np.delete(Ytest, examples, 0)

                    r, cm_list = run_active_learning(classifier, X0, Y0,  Xpool, Ypool,
                                            Xtest, Ytest, query_strategies,
                                            config.query_size, config.budget,
                                            scoring, f_clf_name, config.hide_class)
                    handcraft_cm_lists[f_clf_name] = cm_list
                    Results.update(r)

                    r_new, cm_list_new = run_active_learning(classifier, X0_new, Y0_new,  Xpool, Ypool,
                                            Xtest, Ytest, query_strategies,
                                            config.query_size, config.budget,
                                            scoring, f_clf_name, config.hide_class)
                    handcraft_cm_lists[f_clf_name] = cm_list
                    Results_new.update(r_new)
            else:
                r = run_active_learning(classifier, X0, Y0,  Xpool, Ypool, Xtest, Ytest,
                                        query_strategies, config.query_size, config.budget, scoring, classifier_name)
                Results.update(r)
        joblib.dump(handcraft_cm_lists, save_cm + '.handcraft_cm_lists.pkl')

        if config.train_whole_dataset:
            ###Whole train dataset###
            Xf = np.vstack((X0, Xpool))
            Yf = np.hstack((Y0, Ypool))
            for classifier_name, classifier, param_grid in base_classifiers:
                classifier = GridSearchCV(classifier, param_grid, scoring='f1_macro', n_jobs=-1,
                                          cv=gridsearch_sampler)
                r = run_active_learning(classifier, Xf, Yf,  None, None, Xtest, Ytest,
                                        {'random': query_strategies['random']}, 1, 0, scoring, classifier_name+" (alldata)")
                Results.update(r)

    ###Saving results###
    results_asmatrix = []
    for classif_name, result in Results.items():
        print("===%s===" % classif_name)
        queried_samples = result['queried samples']
        for rname, rs in result.items():
            if(rname.startswith('test_') or 'time' in rname):
                if(rname.startswith('test_')):
                    metric_name = rname.split('_', 1)[-1]
                else:
                    metric_name = rname
                print("%s: %f" % (metric_name, rs[-1]))
                for i, r in enumerate(rs):
                    results_asmatrix.append((classif_name, metric_name, i, queried_samples[i], r))

###Saving results###
    print("WITH TEST EXAMPLES IN TRAIN:")
    results_asmatrix = []
    for classif_name, result in Results_new.items():
        print("===%s===" % classif_name)
        queried_samples = result['queried samples']
        for rname, rs in result.items():
            if(rname.startswith('test_') or 'time' in rname):
                if(rname.startswith('test_')):
                    metric_name = rname.split('_', 1)[-1]
                else:
                    metric_name = rname
                print("%s: %f" % (metric_name, rs[-1]))
                for i, r in enumerate(rs):
                    results_asmatrix.append((classif_name, metric_name, i, queried_samples[i], r))

    if config.save_file is not None:
        df = pd.DataFrame(results_asmatrix,
                          columns=['classifier name', 'metric name', 'step', 'train size', 'value'])
        df.to_csv(config.save_file, index=False)
    rmtree(PIPELINE_CACHE_DIR)
    rmtree(DEEP_CACHE_DIR)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True, help="YAML config file")
    parser.add_argument('-i', '--inputdata', type=Path, required=True, help="Input directory of dataset")
    parser.add_argument('-o', '--outfile', type=Path, required=True, help="Output csv file containing all the results.")
    args = parser.parse_args()
    config = load_yaml(args.config, args.inputdata, args.outfile)
    CONFIG = config
    D = loadRPDBCSData(config.dataset_path)
    main(config, D, args.config)
