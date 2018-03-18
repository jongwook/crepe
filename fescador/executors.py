import multiprocess as mp
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor as TPE
from queue import Queue
from typing import Iterable, Callable, Generator
from .utils import close_iterator


def iterate_until_none(f: Callable, count=1) -> Generator:
    seen = 0
    while True:
        item = f()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            seen += 1
            if seen == count:
                break
            else:
                continue
        yield item


class Executor(ABC):
    @abstractmethod
    def execute(self, transformer: Callable, upstream: Iterable) -> Generator:
        pass

    @abstractmethod
    def parallelism(self):
        pass


class CurrentThreadExecutor(Executor):
    def execute(self, transformer: Callable, upstream: Iterable) -> Generator:
        return transformer(upstream)

    def parallelism(self):
        return 1


class BackgroundThreadExecutor(Executor):
    def execute(self, transformer: Callable, upstream: Iterable) -> Generator:
        queue = Queue(maxsize=1)
        background = TPE(1, 'background-executor')
        background.submit(self._work, upstream, transformer, queue)

        iterator = iterate_until_none(queue.get)
        try:
            for output in iterator:
                yield output
        finally:
            close_iterator(iterator)

        background.shutdown()

    @classmethod
    def _work(cls, upstream: Iterable, transformer: Callable, queue: Queue):
        try:
            for output in transformer(upstream):
                queue.put(output)
        except BaseException as e:
            queue.put(e)
        queue.put(None)

    def parallelism(self):
        return 1


class ThreadPoolExecutor(Executor):
    threads: int

    def __init__(self, threads):
        self.threads = threads

    def execute(self, transformer: Callable, upstream: Iterable) -> Generator:
        input_queue = Queue(maxsize=self.threads)
        output_queue = Queue()

        collector = TPE(1, 'thread-pool-executor-collector')
        collector.submit(self._collect, upstream, input_queue)

        workers = TPE(self.threads, 'thread-pool-executor-worker')
        for _ in range(self.threads):
            workers.submit(self._work, input_queue, transformer, output_queue)

        iterator = iterate_until_none(output_queue.get, self.threads)
        try:
            for output in iterator:
                yield output
        finally:
            close_iterator(iterator)

        collector.shutdown()
        workers.shutdown()

    def _collect(self, upstream: Iterable, input_queue: Queue):
        for item in upstream:
            input_queue.put(item)
        for _ in range(self.threads):
            input_queue.put(None)

    @classmethod
    def _work(cls, input_queue: Queue, transformer: Callable, output_queue: Queue):
        try:
            for output in transformer(iterate_until_none(input_queue.get)):
                output_queue.put(output)
        except BaseException as e:
            output_queue.put(e)
        output_queue.put(None)

    def parallelism(self):
        return self.threads


class MultiProcessingExecutor(Executor):
    def __init__(self, processes):
        self.processes = processes

    def execute(self, transformer: Callable, upstream: Iterable) -> Generator:
        manager = mp.Manager()
        manager.register('None', type(None))
        input_queue = manager.Queue(maxsize=self.processes)
        output_queue = manager.Queue()

        collector = TPE(1, 'multiprocessing-executor-collector')
        collector.submit(self._collect, upstream, input_queue)

        with mp.Pool(self.processes) as pool:
            for _ in range(self.processes):
                pool.apply_async(MultiProcessingExecutor._work, args=(input_queue, transformer, output_queue))
            iterator = iterate_until_none(output_queue.get, self.processes)
            try:
                for output in iterator:
                    yield output
            finally:
                close_iterator(iterator)

        collector.shutdown()

    def _collect(self, upstream: Iterable, input_queue: mp.Queue):
        for item in upstream:
            input_queue.put(item)
        for _ in range(self.processes):
            input_queue.put(None)

    @classmethod
    def _work(cls, input_queue: mp.Queue, transformer: Callable, output_queue: mp.Queue):
        try:
            for output in transformer(iterate_until_none(input_queue.get)):
                output_queue.put(output)
        except BaseException as e:
            output_queue.put(e)
        output_queue.put(None)

    def parallelism(self):
        return self.processes
