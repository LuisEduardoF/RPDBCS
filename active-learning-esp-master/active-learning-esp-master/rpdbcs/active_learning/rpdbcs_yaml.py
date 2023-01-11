from pathlib import Path
import yaml

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
import query_functions
from query_functions import queryfunction_wrapper

_query_dict = {
    'bald': query_functions.bald,
    'k_center_greedy': query_functions.k_center_greedy,
    'random': query_functions.random_sampling,
    'entropy': queryfunction_wrapper(entropy_sampling)
}

_attrs = [
    'train_neuralnet',
    'train_single_classifiers',
    'train_whole_dataset',
    'init_train_size',
    'test_size',
    'query_size',
    'budget',
    'query_strategies',
]


class RALConfig:
    def __init__(self, dataset_path, save_file, **kwargs):
        super().__init__()

        # Check invalid keys
        attr_set = set(_attrs)
        kwkeyset = set(kwargs.keys())
        if not (attr_set & kwkeyset == attr_set):
            raise RuntimeError("YAML key error.")

        for key in _attrs:
            value = kwargs[key]
            setattr(self, key, value)

        # query strategies functions
        fs = {i: _query_dict[i] for i in self.query_strategies}
        self.query_strategies = fs

        self.dataset_path = dataset_path
        self.save_file = save_file

    def __repr__(self):
        repr_ = ""
        for i in _attrs:
            repr_ += "%s: %s\n" % (i, repr(getattr(self, i)))
        return repr_


def load_yaml(yaml_file, dataset_path, save_file):
    if isinstance(yaml_file, str):
        file = Path(yaml_file)

    with open(yaml_file) as yaml_stream:
        al_file = yaml.load(yaml_stream, Loader=yaml.FullLoader)
        return RALConfig(dataset_path, save_file, **al_file)

# yaml.SafeLoader.add_constructor('!RALConfig', RALConfig)
