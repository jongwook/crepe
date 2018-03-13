from abc import ABC, abstractmethod
from .executors import *

class Dataset(ABC):
    @classmethod
    def input(cls, sampler, *args, **kwargs):
        return InputDataset(sampler, args, kwargs)

    @classmethod
    def of(cls, *items):


    @abstractmethod
    def _iterable(self):
        pass

    def _executor(self):
        return CurrentThreadExecutor()

    def __iter__(self):
        pass


class InputDataset(Dataset):
    def __init__(self, sampler, *args, **kwargs):
        self.sampler = sampler
        self.args = args
        self.kwargs = kwargs

    def _iterable(self):
        pass



class FilteredDataset(Dataset):
    pass


class MappedDataset(Dataset):
    pass

class FlatMappedDataset(Dataset):
    pass

