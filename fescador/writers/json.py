import os

try:
    import ujson as JSON
except ImportError:
    import json as JSON

class Writer:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, path, num_splits=None, prefix=None):
        # TODO: shuffled executor for deterministic/even splits
        raise NotImplementedError

json=Writer
