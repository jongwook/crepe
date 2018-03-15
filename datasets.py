from fescador import *
import os


def train_dataset(*names) -> Dataset:
    if len(names) == 0:
        names = ['mdbsynth', 'medleydb', 'nsynth-train']
    paths = [os.path.join('data', 'train', name) for name in names]
    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    return Dataset.roundrobin(datasets)


def test_dataset(*names) -> Dataset:
    if len(names) == 0:
        names = ['bach10', 'mir1k', 'nsynth-test', 'nsynth-valid']
    paths = [os.path.join('data', 'test', name) for name in names]
    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    return Dataset.concat(datasets)
