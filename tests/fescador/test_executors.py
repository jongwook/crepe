from threading import current_thread
from multiprocessing import current_process
from fescador.executors import *


def add1(input):
    for n in input:
        yield n + 1


def thread_info(input):
    for _ in input:
        yield current_thread().ident


def process_info(input):
    for _ in input:
        yield current_process().ident


def test_current_thread_executor():
    ex = CurrentThreadExecutor()
    result = list(ex.execute(add1, [1, 2, 3, 4, 5]))
    assert sum(result) == 20
    assert result == [2, 3, 4, 5, 6]

    tid = next(ex.execute(thread_info, [0]))
    assert current_thread().ident == tid


def test_background_thread_executor():
    ex = BackgroundThreadExecutor()
    result = list(ex.execute(add1, [1, 2, 3, 4, 5]))
    assert sum(result) == 20
    assert result == [2, 3, 4, 5, 6]

    tid = next(ex.execute(thread_info, [0]))
    assert current_thread().ident != tid


def test_thread_pool_executor():
    ex = ThreadPoolExecutor(5)
    result = list(ex.execute(add1, [1, 2, 3, 4, 5]))
    assert sum(result) == 20
    result.sort()
    assert result == [2, 3, 4, 5, 6]

    tids = set(ex.execute(thread_info, range(1)))
    assert current_thread().ident not in tids

    pids = set(ex.execute(process_info, range(1)))
    assert len(pids) == 1
    assert current_process().ident in pids


def test_multi_processing_executor():
    ex = MultiProcessingExecutor(5)
    result = list(ex.execute(add1, [1, 2, 3, 4, 5]))
    assert sum(result) == 20
    result.sort()
    assert result == [2, 3, 4, 5, 6]

    pids = set(ex.execute(process_info, range(1)))
    assert len(pids) == 1
    assert current_process().ident not in pids