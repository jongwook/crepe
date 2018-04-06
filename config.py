import os
import argparse
from datetime import datetime
from typing import List
import importlib


def timestamp():
    return datetime.now().isoformat().replace("-", "").replace(":", "").replace("T", "-").split(".")[0][2:]


parser = argparse.ArgumentParser('CREPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', nargs='?', default='crepe',
                    help='name of the model')
parser.add_argument('experiment_name', nargs='?', default=timestamp(),
                    help='a unique identifier string for this run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='an option to disable data augmentation')
parser.add_argument('--optimizer', default='adam',
                    help='the name of Keras optimizer to use')
parser.add_argument('--batch_size', default=32, type=int,
                    help='the mini-batch size')
parser.add_argument('--validation-take', default=4000, type=int,
                    help='how many examples to take from each validation dataset')
parser.add_argument('--model-capacity', default=32, type=int,
                    help='a multiplier to adjust the model capacity')
parser.add_argument('--load-model', default=None,
                    help='when specified, the full model will be loaded from this path')
parser.add_argument('--load-model-weights', default=None,
                    help='when specified, the model weights will be loaded from this path')
parser.add_argument('--save-model', default='model.h5',
                    help='path to save the model on each epoch')
parser.add_argument('--save-model-weights', default=None,
                    help='path to save the model weights on each epoch; supersedes --save-model')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of epochs to train')
parser.add_argument('--steps-per-epoch', default=1000, type=int,
                    help='number of steps in a batch')
parser.add_argument('--tensorboard', default=False, action='store_true',
                    help='when enabled, tensorboard data will be saved under the log directory')

options = vars(parser.parse_args())
log_dir = os.path.join('experiments', options['experiment_name'])
os.makedirs(log_dir, exist_ok=True)

models = importlib.import_module("models")
keras = models.keras
tf = models.tf


def log_path(*components):
    return os.path.join(log_dir, *components)


def build_model() -> keras.Model:
    """returns the Keras model according to the options"""
    if options['load_model']:
        return keras.models.load_model(options['load_model'])
    else:
        model: keras.Model = getattr(models, options['model'])(**options)
        if options['load_model_weights']:
            model.load_weights(options['load_model_weights'])
        return model


def get_default_callbacks() -> List[keras.callbacks.Callback]:
    """returns a list of callbacks that are used by default"""
    cb = keras.callbacks
    result: List[cb.Callback] = [
        cb.CSVLogger(log_path('learning-curve.tsv'), separator='\t'),
    ]

    if options['save_model_weights']:
        result.append(cb.ModelCheckpoint(log_path(options['save_model_weights']), save_weights_only=True))
    elif options['save_model']:
        result.append(cb.ModelCheckpoint(log_path(options['save_model'])))

    if options['tensorboard']:
        result.append(cb.TensorBoard(log_path('tensorboard')))

    return result
