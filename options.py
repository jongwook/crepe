import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', nargs='?', default='crepe')
parser.add_argument('--optimizer', default='adam', dest='optimizer')

options = vars(parser.parse_args())


def build_model():
    """takes the parsed arguments as kwargs and returns the Keras model"""
    import models
    return getattr(models, options['model'])(**options)
