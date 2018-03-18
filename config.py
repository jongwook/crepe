import os
import argparse
from datetime import datetime
from typing import List


def timestamp():
    return datetime.now().isoformat().replace("-", "").replace(":", "").replace("T", "-").split(".")[0][2:]


parser = argparse.ArgumentParser('CREPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', nargs='?', default='crepe',
                    help='name of the model')
parser.add_argument('experiment_name', nargs='?', default=timestamp(),
                    help='a unique identifier string for this run')
parser.add_argument('--optimizer', default='adam', dest='optimizer',
                    help='the name of Keras optimizer to use')
parser.add_argument('--batch_size', default=32, type=int,
                    help='the mini-batch size')
parser.add_argument('--model-capacity', default=32, type=int,
                    help='a multiplier to adjust the model capacity')
parser.add_argument('--load-model', default=None, dest='load_model',
                    help='when specified, the full model will be loaded from this path')
parser.add_argument('--load-model-weights', default=None, dest='load_model_weights',
                    help='when specified, the model weights will be loaded from this path')
parser.add_argument('--save-model', default='model.h5', dest='save_model',
                    help='path to save the model on each epoch')
parser.add_argument('--save-model-weights', default=None, dest='save_model_weights',
                    help='path to save the model weights on each epoch; supersedes --save-model')
parser.add_argument('--epochs', default=100, dest='epochs', type=int,
                    help='number of epochs to train')
parser.add_argument('--steps-per-epoch', default=1000, dest='steps_per_epoch', type=int,
                    help='number of steps in a batch')
parser.add_argument('--tensorboard', default=False, action='store_true', dest='tensorboard',
                    help='when enabled, tensorboard data will be saved under the log directory')

options = vars(parser.parse_args())
log_dir = os.path.join('experiments', options['experiment_name'])
os.makedirs(log_dir, exist_ok=True)


def log_path(*components):
    return os.path.join(log_dir, *components)


def build_model():
    """returns the Keras model according to the options"""
    import models
    if options['load_model']:
        return models.keras.models.load_model(options['load_model'])
    else:
        model: models.keras.models.Model = getattr(models, options['model'])(**options)
        if options['load_model_weights']:
            model.load_weights(options['load_model_weights'])
        return model

