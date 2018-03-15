from threading import current_thread
import pytest
from fescador import Dataset
import numpy as np


def assert_equal(dataset: Dataset, target: list):
    assert list(dataset) == target


def assert_equal_unordered(dataset: Dataset, target: list):
    assert set(dataset) == set(target)


@pytest.mark.parametrize('options', [{}, {'background': True}])
def test_simple_operations(options):
    data = Dataset([1, 2, 3])

    assert_equal(data, [1, 2, 3])
    assert_equal(data.map(lambda x: x+1, **options), [2, 3, 4])
    assert_equal(data.filter(lambda x: x % 2 == 1, **options), [1, 3])
    assert_equal(data.flatmap(lambda x: [x, x + 1], **options), [1, 2, 2, 3, 3, 4])
    assert_equal(data.filter(lambda x: x % 2 == 0, **options).map(lambda x: x / 2), [1])


def test_executor_options():
    data = Dataset([1, 2, 3])

    assert_equal(data.map(lambda x: x + 1, background=True), [2, 3, 4])
    assert_equal_unordered(data.map(lambda x: x + 1, num_threads=3), [2, 3, 4])
    assert_equal_unordered(data.map(lambda x: x + 1, num_processes=3), [2, 3, 4])

    tid = current_thread().ident
    assert Dataset([1]).map(lambda _: current_thread().ident).first() == tid
    assert Dataset([1]).map(lambda _: current_thread().ident, num_threads=1).first() != tid


@pytest.mark.parametrize('options', [{}, {'background': True}, {'num_threads': 3}, {'num_processes': 3}])
def test_numpy(options):

    data = list(Dataset(np.ones((2, 2))).map(lambda x: x + 1, **options))

    assert np.array_equal(data[0], np.array([2, 2]))
    assert np.array_equal(data[1], np.array([2, 2]))


def test_sequential_mux():
    a = Dataset([1, 2, 3])
    b = Dataset([4, 5, 6])
    mux = Dataset.concat([a, b])
    assert_equal(mux, [1, 2, 3, 4, 5, 6])


def test_round_robin_mux():
    a = Dataset([1, 2, 3])
    b = Dataset([4, 5, 6])
    mux = Dataset.roundrobin([a, b])
    assert_equal(mux, [1, 4, 2, 5, 3, 6])


def test_eratosthenes():
    def integers(starting=2):
        while True:
            yield starting
            starting += 1

    def sieve(dataset: Dataset):
        first = dataset.first()
        tail = dataset.skip(1).filter(lambda x: x % first != 0)
        return first + Dataset(lambda: sieve(tail))

    dataset = Dataset(lambda: integers())
    primes = list(sieve(dataset).take(10))
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
