from threading import current_thread

from fescador import Dataset


def assert_equal(dataset: Dataset, target: list):
    assert list(dataset) == target


def assert_equal_unordered(dataset: Dataset, target: list):
    collected = list(dataset)
    collected.sort()
    target.sort()

    assert collected == target


def test_simple_operations():
    data = Dataset([1, 2, 3])

    assert_equal(data, [1, 2, 3])
    assert_equal(data.map(lambda x: x+1), [2, 3, 4])
    assert_equal(data.filter(lambda x: x % 2 == 1), [1, 3])
    assert_equal(data.flatmap(lambda x: [x, x + 1]), [1, 2, 2, 3, 3, 4])
    assert_equal(data.filter(lambda x: x % 2 == 0).map(lambda x: x / 2), [1])


def test_executor_options():
    data = Dataset([1, 2, 3])

    assert_equal(data.map(lambda x: x + 1, background=True), [2, 3, 4])
    assert_equal_unordered(data.map(lambda x: x + 1, num_threads=3), [2, 3, 4])
    # assert_equal_unordered(data.map(lambda x: x + 1, num_processes=3), [2, 3, 4])

    tid = current_thread().ident
    assert Dataset([1]).map(lambda _: current_thread().ident).take(1) == [tid]
    assert Dataset([1]).map(lambda _: current_thread().ident, num_threads=1).take(1) != [tid]
