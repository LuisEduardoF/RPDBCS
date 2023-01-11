from typing import List
import numpy as np
import sys
from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator


class KindaStratifiedShuffleSplit(StratifiedShuffleSplit):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)
        
    def split(self, X, y, groups=None):
        _, unique_count = np.unique(y, return_counts=True)
        count_one = any([x == 1 for x in unique_count])
        # import ipdb; ipdb.set_trace()
        if not count_one:
            print(f'Using StratifiedShuffleSplit. {unique_count}',
                  file=sys.stderr)
            try:
                foo = [i for i in super().split(X, y, groups)]
                return foo
            except Exception:
                print("But the other one.", file=sys.stderr)
                kwargs = {
                    **self.kwargs,
                    'test_size': 0.5,
                }
                splitter = StratifiedShuffleSplit(**kwargs)
                foo = [i for i in splitter.split(X, y, groups)]
                return foo
        else:
            print(f'Not using StratifiedShuffleSplit. {unique_count}', file=sys.stderr)


        unique_label = np.where(unique_count == 1)[0][0]
        idx_others = np.where(y != unique_label)[0]
        X_ = X[y != unique_label]
        y_ = y[y != unique_label]

        try:
            foo = []
            for idx, (train_index, test_index) in enumerate(super().split(X_, y_, groups)):
                train_index = idx_others[train_index]
                test_index = idx_others[test_index]

                if idx == 0:
                    unique_idx = np.where(y == unique_label)[0][0]
                    train_index = np.append(train_index, unique_idx)

                foo.append((train_index, test_index))
            return foo
        except Exception:
            # import ipdb; ipdb.set_trace()
            print("But the other one.", file=sys.stderr)
            kwargs = {
                **self.kwargs,
                'test_size': 0.5,
            }
            splitter = StratifiedShuffleSplit(**kwargs)
            folds = list(splitter.split(X_, y_, groups))
            unique_idx = np.where(y == unique_label)[0][0]
            folds[0] = (np.append(folds[0][0], unique_idx), folds[0][1])
            return folds


class ConfigTrainSplitter(BaseCrossValidator):
    def __init__(self, train_init_config: List[int]):
        super().__init__()
        assert len(train_init_config) > 0
        self.train_config: List[int] = train_init_config
        self.generator = np.random.default_rng(0)

    def split(self, X, y=None, groups=None):
        assert y is not None

        train_indices = []
        test_indices = []
        for n_class, n_examples in enumerate(self.train_config):
            class_indices = np.where(y == n_class)[0]
            class_idx_test = self.generator.choice(class_indices, n_examples)
            test_indices.extend(class_idx_test)
        test_indices = np.array(test_indices)
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

        yield (train_indices, test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
