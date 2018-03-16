import os

from mir_eval.melody import hz2cents
import numpy as np
from scipy.stats import norm

from fescador import *

classifier_lowest_hz = 31.70
classifier_lowest_cent = hz2cents(np.array([classifier_lowest_hz]))[0]
classifier_cents_per_bin = 20
classifier_octaves = 6
classifier_total_bins = int((1200 / classifier_cents_per_bin) * classifier_octaves)
classifier_cents = np.linspace(0, (classifier_total_bins - 1) * classifier_cents_per_bin, classifier_total_bins) + classifier_lowest_cent
classifier_cents_2d = np.expand_dims(classifier_cents, axis=1)
classifier_norm_stdev = 25
classifier_pdf_normalizer = norm.pdf(0)


def to_classifier_label(pitch, crossentropy=False):
    """
    Converts pitch labels in cents, to a vector representing the classification label
    Uses the normal distribution centered at the pitch and the standard deviation of 25 cents,
    normalized so that the exact prediction has the value 1.0.
    :param pitch: a number of ndarray of dimension 1
    pitch values in cents, as returned by hz2cents with base_frequency = 10 (default)
    :return: ndarray
    """
    if np.isscalar(pitch) or pitch.shape == (1,):
        result = norm.pdf((classifier_cents - pitch) / classifier_norm_stdev)
        result /= crossentropy and classifier_pdf_normalizer or np.sum(result)
    else:
        result = np.zeros((classifier_total_bins, len(pitch)))
        for i, p in enumerate(pitch):
            vec = norm.pdf((classifier_cents - p) / classifier_norm_stdev)
            vec /= crossentropy and classifier_pdf_normalizer or np.sum(vec)
            result[:, i] = vec
    if any(np.isnan(result)):
        result = np.zeros(result.shape)
    return result


def train_dataset(*names, batch_size=32, loop=True) -> Dataset:
    if len(names) == 0:
        names = ['mdbsynth', 'medleydb', 'nsynth-train']

    paths = [os.path.join('data', 'train', name) for name in names]

    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

    if loop:
        datasets = [dataset.loop() for dataset in datasets]

    result = Dataset.roundrobin(datasets)
    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))

    if batch_size:
        result = result.batch(batch_size)

    return result


def validation_dataset(*names) -> Dataset:
    if len(names) == 0:
        names = ['bach10', 'mir1k', 'rwcsynth', 'nsynth-test', 'nsynth-valid']
    paths = [os.path.join('data', 'test', name) for name in names]
    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    result = Dataset.concat(datasets)
    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))
    return result
