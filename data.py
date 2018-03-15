from fescador import *
import os


def train_dataset(*names) -> Dataset:
    if len(names) == 0:
        names = ['mdbsynth', 'medleydb', 'nsynth-train']
    datasets = [Dataset.read.tfrecord(os.path.join('data', 'train', name)) for name in names]
    return Dataset.roundrobin(datasets)


def test_dataset(*names) -> Dataset:
    if len(names) == 0:
        names = ['bach10', 'mir1k', 'nsynth-test', 'nsynth-valid']
    datasets = [Dataset.read.tfrecord(os.path.join('data', 'test', name)) for name in names]
    return Dataset.concat(datasets)
