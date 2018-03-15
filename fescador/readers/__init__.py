from importlib import import_module


class LazyLoader:
    def __getattr__(self, name):
        if name in ['tfrecord']:
            module = import_module('.' + name, package=__package__)
            return getattr(module, name)
        else:
            raise AttributeError('Unknown attribute: {}'.format(name))
