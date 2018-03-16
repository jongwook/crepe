def close_iterator(iterator):
    if hasattr(iterator, 'close'):
        iterator.close()
