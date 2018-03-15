from importlib import import_module


class LazyLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        if name in ['tfrecord', 'json']:
            module = import_module('.' + name, package=__package__)
            writer = getattr(module, name)
            return writer(self.dataset)
        else:
            raise AttributeError('Unknown attribute: {}'.format(name))
