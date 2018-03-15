from fescador import *

from test_datasets import assert_equal, assert_equal_unordered


def test_sequential_mux():
    a = Dataset([1, 2, 3])
    b = Dataset([4, 5, 6])
    mux = SequentialMux([a, b])
    assert_equal(mux, [1, 2, 3, 4, 5, 6])


def test_round_robin_mux():
    a = Dataset([1, 2, 3])
    b = Dataset([4, 5, 6])
    mux = RoundRobinMux([a, b])
    assert_equal(mux, [1, 4, 2, 5, 3, 6])
