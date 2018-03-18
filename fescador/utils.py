import numpy as np
from numbers import Number


def close_iterator(iterator):
    if hasattr(iterator, 'close'):
        iterator.close()


def make_batch(items, default_dtype=None):
    # case 1: each item is dict of stuff -> group by key
    if all(isinstance(item, dict) for item in items):
        keys = {key for item in items for key in item}
        return {key: make_batch([item[key] for item in items], default_dtype) for key in keys}

    # case 2: each item is tuple or list of stuff -> group by index
    if all(isinstance(item, (tuple, list)) for item in items):
        length = min(*[len(item) for item in items])
        return tuple(make_batch([item[i] for item in items], default_dtype) for i in range(length))

    # case 3: each item is numpy array -> prepend a batch dimension
    if all(isinstance(item, np.ndarray) for item in items):
        return np.vstack(np.expand_dims(array, axis=0) for array in items)

    # case 3: items are numbers -> 1D nparray
    if all(isinstance(item, Number) for item in items):
        return np.array(items, dtype=default_dtype)

    return tuple(items)
