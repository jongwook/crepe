import argparse
import gzip
import os
import sys

import matplotlib.cm
import numpy as np
import scipy.misc
from mir_eval.melody import hz2cents
from numpy.lib.stride_tricks import as_strided
from resampy import resample
from scipy.io import wavfile

from datasets import to_local_average_cents

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='path to the HDF5 file that contains the Keras model')
parser.add_argument('input_path',
                    help='path that contains .wav or .npy.gz files to run the model on')
parser.add_argument('output_path', nargs='?', default=None,
                    help='path to save the prediction and salience results (default: same as input_path)')
parser.add_argument('--save-numpy', action='store_true',
                    help='save the salience representation to .npy file as well')
parser.add_argument('--truth-path', default=None,
                    help='path to the corresponding .csv or .npy.gz files that contains the ground-truth annotations')
args = parser.parse_args()

if args.output_path is None:
    args.output_path = args.input_path


def wav_stream(files):
    for file in files:
        srate, data = wavfile.read(os.path.join(args.input_path, file))
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        if srate != 16000:
            data = resample(data, srate, 16000)
            srate = 16000
        hop_length = int(srate / 100)
        n_frames = 1 + int((len(data) - 1024) / hop_length)
        frames = as_strided(data, shape=(1024, n_frames),
                            strides=(data.itemsize, hop_length * data.itemsize))
        frames = frames.transpose().astype(np.float32)
        yield (file, frames)


def npygz_stream(files):
    for file in files:
        with gzip.open(os.path.join(args.input_path, file)) as f:
            yield (file, np.load(f).transpose())


files = [file for file in os.listdir(args.input_path) if file.lower().endswith('.wav')]
files.sort()
num_files = len(files)
if num_files > 0:
    print(num_files, "wav files found")
    stream = wav_stream(files)
else:
    files = [file for file in os.listdir(args.input_path) if file.lower().endswith('.npy.gz')]
    files.sort()
    num_files = len(files)
    if num_files > 0:
        print(num_files, ".npy.gz files found")
        stream = npygz_stream(files)
    else:
        raise ValueError("No .wav or .npy.gz files found in ", args.input_path)


import keras  # noqa

model = keras.models.load_model(args.model)
model.summary()

for name, data in stream:
    print('processing', name, 'of shape', data.shape)
    data -= np.mean(data, axis=1)[:, np.newaxis]
    data /= np.std(data, axis=1)[:, np.newaxis]
    predictions = model.predict(data, verbose=True)
    cents = to_local_average_cents(predictions)
    hertz = 10.0 * 2 ** (cents / 1200.0)
    timestamps = 0.01 * np.array(range(hertz.shape[0]))
    result = np.vstack([timestamps, hertz]).transpose()
    result_file = os.path.join(args.output_path, name + '.f0.csv')
    np.savetxt(result_file, result, fmt='%.6f', delimiter=',')

    figure_file = os.path.join(args.output_path, name + '.salience.png')
    dpi = 120
    fig = plt.figure(figsize=(predictions.shape[0] / dpi, predictions.shape[1] / dpi), dpi=dpi)
    fig.figimage(predictions.transpose(), vmin=0, vmax=1, cmap='inferno', origin='lower')
    fig.savefig(figure_file)
    plt.close(fig)

    if args.save_numpy:
        salience_file = os.path.join(args.output_path, name + '.salience.npy')
        np.save(salience_file, predictions)

    if args.truth_path:
        basename = name.replace('.npy.gz', '')
        csv_path = os.path.join(args.truth_path, basename + '.csv')
        npygz_path = os.path.join(args.truth_path, basename + '.npy.gz')
        if os.path.isfile(csv_path):
            truth = np.loadtxt(csv_path)
        elif os.path.isfile(npygz_path):
            with gzip.open(npygz_path) as f:
                truth = np.load(f)
        else:
            print('truth file for {} not found'.format(name), file=sys.stderr)
        truth = hz2cents(truth)

        image = scipy.misc.imread(figure_file, mode='RGB')
        image = np.pad(image, [(40, 0), (0, 0), (0, 0)], mode='constant')
        jet = matplotlib.cm.get_cmap('jet')
        for i in range(image.shape[1]):
            if truth[i] < 1:
                continue  # no-voice
            image[:40, i, :] = 255 * np.array(jet(int(abs(truth[i] - cents[i]))))[:3]

        scipy.misc.imsave(figure_file.replace('.png', '.eval.png'), image)