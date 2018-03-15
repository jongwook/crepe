from random import Random
from typing import List

from . import readers
from . import writers
from .executors import *


class BaseDataset(ABC):
    def map(self, function, **kwargs) -> 'BaseDataset':
        return MappedDataset(self, function, **kwargs)

    def flatmap(self, function, **kwargs) -> 'BaseDataset':
        return FlatMappedDataset(self, function, **kwargs)

    def filter(self, function, **kwargs) -> 'BaseDataset':
        return FilteredDataset(self, function, **kwargs)

    def transform(self, function, **kwargs) -> 'BaseDataset':
        return TransformedDataset(self, function, **kwargs)

    def foreach(self, function, **kwargs) -> None:
        for _ in self.map(function, **kwargs):
            pass

    def collect(self) -> list:
        return list(self)

    def take(self, count):
        return [item for item, n in zip(self, range(count))]

    def skip(self, count) -> 'BaseDataset':
        raise NotImplementedError

    def group(self, size) -> 'BaseDataset':
        raise NotImplementedError

    def batch(self, size) -> 'BaseDataset':
        raise NotImplementedError

    def cache(self) -> 'BaseDataset':
        raise NotImplementedError

    def select(self, *keys, **kwargs):
        return self.map(lambda row: {key: row[key] for key in keys}, **kwargs)

    def shuffle(self, buffer_size, seed=None):
        def generator(buffer_size, seed):
            random = Random(seed)

            # fill the buffer
            iterator = iter(self)
            buffer = [item for _, item in zip(range(buffer_size), iterator)]
            random.shuffle(buffer)

            # sample one from the buffer and replace with a new one pulled from the iterator
            for item in iterator:
                i = random.randrange(buffer_size)
                yield buffer[i]
                buffer[i] = item

            # drain any remaining items
            for item in buffer:
                if item is not None:
                    yield item

        return self.flatmap

    def executor(self, **kwargs) -> Executor:
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
            return self._upstream()[0].executor()
        except IndexError:
            return CurrentThreadExecutor()

    @abstractmethod
    def _upstream(self) -> List['BaseDataset']:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class Dataset(BaseDataset):
    read = readers.LazyLoader()
    write: writers.LazyLoader

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

        self.write = writers.LazyLoader(self)

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


class TransformedDataset(BaseDataset):
    def __init__(self, parent, transformer, **kwargs):
        self._parent = parent
        self._transformer = transformer
        self._kwargs = kwargs

    def _upstream(self):
        return [self._parent]

    def __iter__(self):
        return self.executor(**self._kwargs).execute(self._transformer, self._parent)


class MappedDataset(TransformedDataset):
    def __init__(self, parent, mapper, **kwargs):
        super().__init__(parent, lambda items: (mapper(x) for x in items), **kwargs)


class FilteredDataset(TransformedDataset):
    def __init__(self, parent, predicate, **kwargs):
        super().__init__(parent, lambda items: (x for x in items if predicate(x)), **kwargs)


class FlatMappedDataset(TransformedDataset):
    def __init__(self, parent, flatmapper, **kwargs):
        super().__init__(parent, lambda items: (y for x in items for y in flatmapper(x)), **kwargs)
