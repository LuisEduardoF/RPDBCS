from pathlib import Path
import yaml

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
import query_functions
from query_functions import queryfunction_wrapper, queryrandom_wrapper, sequential_query, NewFaultQuery, NewFaultWithMargin

_query_dict = {
    'bald': query_functions.bald,
    'k_center_greedy': query_functions.k_center_greedy,
    'random': query_functions.random_sampling,
    'entropy': queryfunction_wrapper(entropy_sampling),
    'random_entropy': sequential_query(
        query_functions.random_sampling,
        queryfunction_wrapper(entropy_sampling)
    ),
    'entropy_random': sequential_query(
        queryfunction_wrapper(entropy_sampling),
        query_functions.random_sampling
    ),
    'top margin': queryfunction_wrapper(uncertainty_sampling),
    '1-2 margin': queryfunction_wrapper(margin_sampling),
    'top-margin_random': sequential_query(
        queryfunction_wrapper(uncertainty_sampling),
        query_functions.random_sampling
    ),
    '1-2-margin_random': sequential_query(
        queryfunction_wrapper(margin_sampling),
        query_functions.random_sampling
    ),
    'random_top-margin': sequential_query(
        query_functions.random_sampling,
        queryfunction_wrapper(uncertainty_sampling)
    ),
    'random_1-2-margin': sequential_query(
        query_functions.random_sampling,
        queryfunction_wrapper(margin_sampling)
    ),
    'k_center_greedy_entropy': sequential_query(
        query_functions.k_center_greedy,
        queryfunction_wrapper(entropy_sampling)
    ),
    'new_fault-1': NewFaultQuery(1),
    'new_fault-2': NewFaultQuery(2),
    'new_fault-3': NewFaultQuery(3),
    'new_fault-4': NewFaultQuery(4),
    'entropy-new_fault-1': sequential_query(
        queryfunction_wrapper(entropy_sampling),
        NewFaultQuery(1)
    ),
    'entropy-new_fault-2': sequential_query(
        queryfunction_wrapper(entropy_sampling),
        NewFaultQuery(2)
    ),
    'entropy-new_fault-3': sequential_query(
        queryfunction_wrapper(entropy_sampling),
        NewFaultQuery(3)
    ),
    'entropy-new_fault-4': sequential_query(
        queryfunction_wrapper(entropy_sampling),
        NewFaultQuery(4)
    ),
    'new_fault_12margin-1': NewFaultWithMargin(1),
    'new_fault_12margin-2': NewFaultWithMargin(2),
    'new_fault_12margin-3': NewFaultWithMargin(3),
    'new_fault_12margin-4': NewFaultWithMargin(4),
}

_attrs = [
    'train_neuralnet',
    'train_single_classifiers',
    'train_whole_dataset',
    'init_train_size',
    'init_train_config',
    'init_pool_max_config',  # Max number of examples of class in pool
    'test_size',
    'query_size',
    'budget',
    'query_strategies',
    'kfolds',      # None means 'don't use folds'
    'base_clf',    # Sklearn classifiers
    'hide_class',  # None means 'don't hide any class'
]


class RALConfig:
    def __init__(self, dataset_path, save_file, **kwargs):
        super().__init__()

        # Check invalid keys
        attr_set = set(_attrs)
        kwkeyset = set(kwargs.keys())
        if attr_set & kwkeyset != attr_set:
            diff = attr_set - kwkeyset
            if len(diff) <= 0:
                raise RuntimeError("YAML key error.")

            if not diff.issubset({'init_pool_max_config'}):
                raise RuntimeError("YAML key error.")

        for key in _attrs:
            try:
                value = kwargs[key]
                setattr(self, key, value)
            except KeyError:
                pass

        # assert len(self.init_train_config) == 5
        # assert not all(map(lambda x: x <= 0, self.init_train_config))

        # query strategies functions
        fs = {i: _query_dict[i] for i in self.query_strategies}
        self.query_strategies = fs

        self.dataset_path = dataset_path
        self.save_file = save_file

    def __repr__(self):
        repr_ = ""
        for i in _attrs:
            try:
                repr__ = "%s: %s\n" % (i, repr(getattr(self, i)))
            except Exception:
                repr__ = ""
            repr_ += repr__
        return repr_


def load_yaml(yaml_file, dataset_path, save_file):
    if isinstance(yaml_file, str):
        file = Path(yaml_file)

    with open(yaml_file) as yaml_stream:
        al_file = yaml.load(yaml_stream, Loader=yaml.FullLoader)
        return RALConfig(dataset_path, save_file, **al_file)

# yaml.SafeLoader.add_constructor('!RALConfig', RALConfig)
