import numpy as np
from numbers import Number

def close_iterator(iterator):
    if hasattr(iterator, 'close'):
        iterator.close()


def make_minibatch(items):
    # case 1: dict of stuff -> group by key
    if all(isinstance(item, dict) for item in items):
        keys = {key for item in items for key in item}
        return {key: make_minibatch([item[key] for item in items]) for key in keys}

    # case 2: just numpy arrays -> prepend a batch dimension
    if all(isinstance(item, np.ndarray) for item in items):
        return np.vstack(np.expand_dims(array, axis=0) for array in items)

    # case 3: numbers -> 1D nparray
    if all(isinstance(item, Number) for item in items):
        return np.array(items)

    return tuple(items)
