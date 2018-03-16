def close_iterator(iterator):
    if hasattr(iterator, 'close'):
        iterator.close()


class TransformerGuard:
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, upstream):
        iterator = iter(upstream)
        try:
            for item in self.transformer(iterator):
                yield item
        except GeneratorExit:
            close_iterator(iterator)
            raise
