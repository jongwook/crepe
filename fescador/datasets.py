from typing import List

import numpy as np
import pandas as pd

from . import readers
from . import writers
from .executors import *
from .utils import *


class Dataset(ABC):
    read = readers.LazyLoader()
    write: writers.LazyLoader

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls is Dataset and InMemoryDataset or cls)

    def __init__(self, *args, **kwargs):
        self.write = writers.LazyLoader(self)
        self._shape = None

    def map(self, function, **executor_config) -> 'Dataset':
        return MappedDataset(self, function, **executor_config)

    def flatmap(self, function, **executor_config) -> 'Dataset':
        return FlatMappedDataset(self, function, **executor_config)

    def filter(self, function, **executor_config) -> 'Dataset':
        return FilteredDataset(self, function, **executor_config)

    def transform(self, function, **executor_config) -> 'Dataset':
        return TransformedDataset(self, function, **executor_config)

    def foreach(self, function, **executor_config) -> None:
        for _ in self.map(lambda item: function(item) or True, **executor_config):
            pass

    def collect(self) -> list:
        return list(self)

    def shape(self, template=None):
        root = template is None
        if root:
            if hasattr(self, '_shape') and self._shape is not None:
                return self._shape
            template = self.first()

        if isinstance(template, list):
            shape = len(template)
        elif isinstance(template, np.ndarray):
            shape = template.shape
        elif isinstance(template, tuple):
            shape = tuple(self.shape(item) for item in template)
        elif isinstance(template, dict):
            shape = {key: self.shape(template[key]) for key in template}
        else:
            shape = tuple()

        if root:
            self._shape = shape

        return shape

    def first(self):
        try:
            return self.take(1)[0]
        except IndexError:
            raise ValueError('empty dataset')

    def take(self, count: int) -> list:
        iterator = iter(self)
        result = [item for item, n in zip(iterator, range(count))]
        close_iterator(iterator)
        return result

    def skip(self, count: int) -> 'Dataset':
        return self.transform(lambda items: (item for i, item in enumerate(self) if i >= count), background=False)

    def loop(self, count: int=-1) -> 'Dataset':
        return LoopedDataset(self, count)

    def group(self, size) -> 'Dataset':
        raise NotImplementedError

    def batch(self, size) -> 'Dataset':
        raise NotImplementedError

    def cache(self) -> 'Dataset':
        return CachedDataset(self)

    def select(self, *keys, **executor_config):
        return self.map(lambda row: {key: row[key] for key in keys}, **executor_config)

    def select_tuple(self, *keys, **executor_config):
        return self.map(lambda row: tuple(row[key] for key in keys), **executor_config)

    def shuffle(self, buffer_size=-1, seed=None):
        raise NotImplementedError

    @classmethod
    def concat(cls, datasets: List['Dataset']) -> 'Dataset':
        from .mux import SequentialMux
        return SequentialMux(datasets)

    @classmethod
    def roundrobin(cls, datasets: List['Dataset']) -> 'Dataset':
        from .mux import RoundRobinMux
        return RoundRobinMux(datasets)

    def __add__(self, other) -> 'Dataset':
        if isinstance(other, Dataset) or callable(other):
            from .mux import SequentialMux
            return SequentialMux([self, other])
        raise ValueError("unknown operand type: {}".format(type(other)))

    def __radd__(self, other) -> 'Dataset':
        return PrependedDataset(self, other)

    def executor(self, **executor_config) -> Executor:
        if 'executor' in executor_config:
            executor = executor_config['executor']
            if isinstance(executor, Executor):
                return executor
        if 'background' in executor_config:
            if executor_config['background'] is True:
                return BackgroundThreadExecutor()
            else:
                return CurrentThreadExecutor()
        if 'num_threads' in executor_config:
            return ThreadPoolExecutor(int(executor_config['num_threads']))
        if 'num_processes' in executor_config:
            return MultiProcessingExecutor(int(executor_config['num_processes']))
        if len(executor_config) > 0:
            raise ValueError("Unknown executor_config:", executor_config.keys())
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

    def __repr__(self):
        return "({} of shape {})".format(type(self).__name__, self.shape())


class InMemoryDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if (len(args) == 0) == (len(kwargs) == 0):
            raise ValueError('either args or kwargs must be given')

        if len(args) == 1:
            if isinstance(args[0], pd.DataFrame):
                # pandas dataframe to dataset of dicts
                self._items = args[0].to_dict('records')
            else:
                # dataset of single items
                self._items = args[0]
        elif len(args) > 0:
            # dataset of tuples:
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
            for record in zip(*values):
                yield {key: value for key, value in zip(keys, record)}
        else:
            items = callable(self._items) and self._items() or self._items
            iterator = iter(items)
            try:
                for item in iterator:
                    yield item
            except GeneratorExit:
                close_iterator(iterator)
                raise


class LoopedDataset(Dataset):
    def __init__(self, upstream, count=-1):
        super().__init__()
        self.upstream = upstream
        self.count = count

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        for _ in self.count >= 0 and range(self.count) or iter(int, 1):
            iterator = iter(self.upstream)
            try:
                for item in iterator:
                    yield item
            except GeneratorExit:
                close_iterator(iterator)
                raise


class TransformedDataset(Dataset):
    class TransformerGuard:
        def __init__(self, transformer):
            self.transformer = transformer

        def __call__(self, upstream):
            iterator = iter(upstream)
            try:
                for item in self.transformer(iterator):
                    yield item
            except GeneratorExit:
                close_iterator(iterator)
                raise

    def __init__(self, upstream, transformer, **executor_config):
        super().__init__()
        self.upstream = upstream
        self.transformer = transformer
        self.executor_config = executor_config

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        return self.executor(**self.executor_config).execute(self.TransformerGuard(self.transformer), self.upstream)


class MappedDataset(TransformedDataset):
    def __init__(self, upstream, mapper, **executor_config):
        super().__init__(upstream, lambda items: (mapper(x) for x in items), **executor_config)


class FilteredDataset(TransformedDataset):
    def __init__(self, upstream, predicate, **executor_config):
        super().__init__(upstream, lambda items: (x for x in items if predicate(x)), **executor_config)


class FlatMappedDataset(TransformedDataset):
    def __init__(self, upstream, flatmapper, **executor_config):
        super().__init__(upstream, lambda items: (y for x in items for y in flatmapper(x)), **executor_config)


class CachedDataset(Dataset):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream
        self.cache = list(upstream)

    def _upstream(self):
        return self.upstream

    def __iter__(self):
        return iter(self.cache)


class PrependedDataset(Dataset):
    def __init__(self, upstream, other):
        super().__init__()
        self.upstream = upstream
        self.other = other

    def _upstream(self) -> List['Dataset']:
        return [self.upstream]

    def __iter__(self):
        yield self.other
        iterator = iter(self.upstream)
        try:
            for item in self.upstream:
                yield item
        except GeneratorExit:
            close_iterator(iterator)
            raise
