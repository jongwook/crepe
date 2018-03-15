from random import Random
from typing import List

from . import readers
from . import writers
from .executors import *


class Dataset(ABC):
    read = readers.LazyLoader()
    write: writers.LazyLoader

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls is Dataset and InMemoryDataset or cls)

    def __init__(self, *args, **kwargs):
        self.write = writers.LazyLoader(self)

    def map(self, function, **kwargs) -> 'Dataset':
        return MappedDataset(self, function, **kwargs)

    def flatmap(self, function, **kwargs) -> 'Dataset':
        return FlatMappedDataset(self, function, **kwargs)

    def filter(self, function, **kwargs) -> 'Dataset':
        return FilteredDataset(self, function, **kwargs)

    def transform(self, function, **kwargs) -> 'Dataset':
        return TransformedDataset(self, function, **kwargs)

    def foreach(self, function, **kwargs) -> None:
        for _ in self.map(lambda item: function(item) or True, **kwargs):
            pass

    def collect(self) -> list:
        return list(self)

    def first(self):
        return next(iter(self))

    def take(self, count: int) -> 'Dataset':
        return self.transform(lambda items: (item for item, n in zip(self, range(count))), background=False)

    def skip(self, count: int) -> 'Dataset':
        return self.transform(lambda items: (item for i, item in enumerate(self) if i >= count), background=False)

    def loop(self, count: int=-1) -> 'Dataset':
        return LoopedDataset(self, count)

    def group(self, size) -> 'Dataset':
        raise NotImplementedError

    def batch(self, size) -> 'Dataset':
        raise NotImplementedError

    def cache(self) -> 'Dataset':
        raise NotImplementedError

    def select(self, *keys, **kwargs):
        return self.map(lambda row: {key: row[key] for key in keys}, **kwargs)

    def shuffle(self, buffer_size, seed=None):
        raise NotImplementedError

    def executor(self, **kwargs) -> Executor:
        if 'executor' in kwargs:
            executor = kwargs['executor']
            if isinstance(executor, Executor):
                return executor
        if 'background' in kwargs:
            if kwargs['background'] is True:
                return BackgroundThreadExecutor()
            else:
                return CurrentThreadExecutor()
        if 'num_threads' in kwargs:
            return ThreadPoolExecutor(int(kwargs['num_threads']))
        if 'num_processes' in kwargs:
            return MultiProcessingExecutor(int(kwargs['num_processes']))
        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs:", kwargs.keys())
        try:
            return self._upstream()[0].executor()
        except IndexError:
            return CurrentThreadExecutor()

    @abstractmethod
    def _upstream(self) -> List['Dataset']:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class InMemoryDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

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


class LoopedDataset(Dataset):
    def __init__(self, upstream, count=-1):
        super().__init__()
        self.upstream = upstream
        self.count = count

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        for _ in self.count >= 0 and range(self.count) or iter(int, 1):
            for item in self.upstream:
                yield item


class TransformedDataset(Dataset):
    def __init__(self, upstream, transformer, **kwargs):
        super().__init__()
        self.upstream = upstream
        self.transformer = transformer
        self.kwargs = kwargs

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        return self.executor(**self.kwargs).execute(self.transformer, self.upstream)


class MappedDataset(TransformedDataset):
    def __init__(self, upstream, mapper, **kwargs):
        super().__init__(upstream, lambda items: (mapper(x) for x in items), **kwargs)


class FilteredDataset(TransformedDataset):
    def __init__(self, upstream, predicate, **kwargs):
        super().__init__(upstream, lambda items: (x for x in items if predicate(x)), **kwargs)


class FlatMappedDataset(TransformedDataset):
    def __init__(self, upstream, flatmapper, **kwargs):
        super().__init__(upstream, lambda items: (y for x in items for y in flatmapper(x)), **kwargs)
