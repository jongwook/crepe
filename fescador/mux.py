from random import Random
from typing import List

from .datasets import Dataset
from .executors import Executor, CurrentThreadExecutor


class Mux(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def _upstream(self) -> List[Dataset]:
        return self.datasets

    def executor(self, **kwargs) -> Executor:
        return CurrentThreadExecutor()


class SequentialMux(Mux):
    def __iter__(self):
        for dataset in self.datasets:
            if callable(dataset):
                dataset = dataset()
            for item in dataset:
                yield item


class RoundRobinMux(Mux):
    def __iter__(self):
        datasets = [callable(d) and d() or d for d in self.datasets]
        iterators = list(map(iter, datasets))
        while len(iterators) > 0:
            for iterator in iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterators.remove(iterator)


class PoissonMux(Mux):
    def __init__(self, datasets):
        super().__init__(datasets)
        raise NotImplementedError  # TODO

    def __iter__(self):
        raise NotImplementedError  # TODO
