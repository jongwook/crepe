from abc import ABC, abstractmethod
from .executors import *
from typing import List
from . import readers


class BaseDataset(ABC):
    def map(self, function, **kwargs) -> 'BaseDataset':
        return MappedDataset(self, function, **kwargs)

    def flatmap(self, function, **kwargs) -> 'BaseDataset':
        return FlatMappedDataset(self, function, **kwargs)

    def filter(self, function, **kwargs) -> 'BaseDataset':
        return FilteredDataset(self, function, **kwargs)

    def foreach(self, function, **kwargs):
        for _ in self.map(function, **kwargs):
            pass

    def collect(self):
        return list(self)

    def take(self, count):
        return [item for item, n in zip(self, range(count))]

    def _executor(self, **kwargs) -> Executor:
        if 'executor' in kwargs:
            executor = kwargs['executor']
            if isinstance(executor, Executor):
                return executor
        if 'background' in kwargs:
            return BackgroundThreadExecutor()
        if 'num_threads' in kwargs:
            return ThreadPoolExecutor(int(kwargs['num_threads']))
        if 'num_processes' in kwargs:
            return MultiProcessingExecutor(int(kwargs['num_processes']))
        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs:", kwargs.keys())
        try:
            return self._upstream()[0]._executor()
        except IndexError:
            return CurrentThreadExecutor()

    @abstractmethod
    def _upstream(self) -> List['BaseDataset']:
        pass

    @abstractmethod
    def __iter__(self):
        pass


class Dataset(BaseDataset):
    read = readers.LazyLoader()

    def __init__(self, *args, **kwargs):
        if (len(args) == 0) == (len(kwargs) == 0):
            raise ValueError('either args or kwargs must be given')

        # dataset of tuples (or single items):
        if len(args) > 0:
            if len(args) == 1:
                self._items = args[0]
            else:
                self._items = zip(*args)

        # dataset of dicts
        if len(kwargs) > 0:
            self._items = kwargs

    def _upstream(self):
        return []

    def __iter__(self):
        if isinstance(self._items, dict):
            keys = list(self._items.keys())
            values = [self._items[key] for key in keys]
            for tuple in zip(*values):
                yield {key: value for key, value in zip(keys, tuple)}
        else:
            for item in self._items:
                yield item


class InMemoryDataset(BaseDataset):
    def __init__(self, items):
        self._items = items

    def _upstream(self):
        return []

    def __iter__(self):
        return iter(self._items)


class FlatMappedDataset(BaseDataset):
    def __init__(self, parent, function, **kwargs):
        self._parent = parent
        self._function = function
        self._kwargs = kwargs

    def _upstream(self):
        return [self._parent]

    def __iter__(self):
        return self._executor(**self._kwargs).execute(self._function, self._parent)


class FilteredDataset(FlatMappedDataset):
    def __init__(self, parent, function, **kwargs):
        super(FilteredDataset, self).__init__(parent, lambda x: function(x) and [x] or [], **kwargs)


class MappedDataset(FlatMappedDataset):
    def __init__(self, parent, function, **kwargs):
        super(MappedDataset, self).__init__(parent, lambda x: [function(x)], **kwargs)


class InputDataset(BaseDataset):
    args = []
    kwargs = {}

    def __init__(self, iterable_supplier, *args, **kwargs):
        if callable(iterable_supplier):
            self.iterable_supplier = iterable_supplier
            self.args = args
            self.kwargs = kwargs
        else:
            self.iterable_supplier = lambda: iterable_supplier
