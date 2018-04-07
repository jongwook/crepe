import os

from flazy import Dataset
from mir_eval.melody import hz2cents
from scipy.stats import norm

from transforms import *

classifier_lowest_hz = 31.70
classifier_lowest_cent = hz2cents(np.array([classifier_lowest_hz]))[0]
classifier_cents_per_bin = 20
classifier_octaves = 6
classifier_total_bins = int((1200 / classifier_cents_per_bin) * classifier_octaves)
classifier_cents = np.linspace(0, (classifier_total_bins - 1) * classifier_cents_per_bin, classifier_total_bins) + classifier_lowest_cent
classifier_cents_2d = np.expand_dims(classifier_cents, axis=1)
classifier_norm_stdev = 25
classifier_pdf_normalizer = norm.pdf(0)


def to_classifier_label(pitch):
    """
    Converts pitch labels in cents, to a vector representing the classification label
    Uses the normal distribution centered at the pitch and the standard deviation of 25 cents,
    normalized so that the exact prediction has the value 1.0.
    :param pitch: a number or numpy array of shape (1,)
    pitch values in cents, as returned by hz2cents with base_frequency = 10 (default)
    :return: ndarray
    """
    result = norm.pdf((classifier_cents - pitch) / classifier_norm_stdev).astype(np.float32)
    result /= classifier_pdf_normalizer
    return result


def to_weighted_average_cents(label):
    if label.ndim == 1:
        productsum = np.sum(label * classifier_cents)
        weightsum = np.sum(label)
        return productsum / weightsum
    if label.ndim == 2:
        productsum = np.dot(label, classifier_cents)
        weightsum = np.sum(label, axis=1)
        return productsum / weightsum
    raise Exception("label should be either 1d or 2d ndarray")


def to_local_average_cents(label):
    if label.ndim == 1:
        argmax = int(np.argmax(label))
        start = max(0, argmax - 4)
        end = min(len(label), argmax + 5)
        label = label[start:end]
        productsum = np.sum(label * classifier_cents[start:end])
        weightsum = np.sum(label)
        return productsum / weightsum
    if label.ndim == 2:
        return np.array([to_local_average_cents(label[i, :]) for i in range(label.shape[0])])
    raise Exception("label should be either 1d or 2d ndarray")


def train_dataset(*names, batch_size=32, loop=True, augment=True) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    paths = [os.path.join('data', 'train', name) for name in names]

    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

    if loop:
        datasets = [dataset.repeat() for dataset in datasets]

    result = Dataset.roundrobin(datasets)
    result = result.starmap(normalize)

    if augment:
        result = result.starmap(add_noise)
        result = result.starmap(pitch_shift)

    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))

    if batch_size:
        result = result.batch(batch_size)

    return result


def validation_dataset(*names, seed=None, take=None) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    paths = [os.path.join('data', 'test', name) for name in names]

    all_datasets = []

    for path in paths:
        files = [os.path.join(path, file) for file in os.listdir(path)]

        if seed:
            files = Random(seed).sample(files, len(files))

        datasets = [Dataset.read.tfrecord(file, compression='gzip') for file in files]
        datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

        if seed:
            datasets = [dataset.shuffle(seed=seed) for dataset in datasets]
        if take:
            datasets = [dataset.take(take) for dataset in datasets]

        all_datasets.append(Dataset.concat(datasets))

    result = Dataset.roundrobin(all_datasets)
    result = result.starmap(normalize)
    result = result.map(lambda x: (x[0], to_classifier_label(hz2cents(x[1]))))

    return result
