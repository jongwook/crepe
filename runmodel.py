import argparse
import gzip
import os

import matplotlib.cm
import numpy as np
from numpy.lib.stride_tricks import as_strided
from resampy import resample
from scipy.io import wavfile

from datasets import to_local_average_cents

matplotlib.use('Agg')

import matplotlib.pyplot as plt  #noqa

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='path to the HDF5 file that contains the Keras model')
parser.add_argument('input_path',
                    help='path that contains .wav or .npy.gz files to run the model on')
parser.add_argument('output_path', nargs='?', default=None,
                    help='path to save the prediction and salience results (default: same as input_path)')
parser.add_argument('--save-numpy', action='store_true',
                    help='save the salience representation to .npy file as well')
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
num_files = len(files)
if num_files > 0:
    print(num_files, "wav files found")
    stream = wav_stream(files)
else:
    files = [file for file in os.listdir(args.input_path) if file.lower().endswith('.npy.gz')]
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
    salience = model.predict(data, verbose=True)
    cents = to_local_average_cents(salience)
    hertz = 10.0 * 2 ** (cents / 1200.0)
    timestamps = 0.01 * np.array(range(hertz.shape[0]))
    result = np.vstack([timestamps, hertz]).transpose()
    result_file = os.path.join(args.output_path, name + '.f0.csv')
    np.savetxt(result_file, result, fmt='%.6f', delimiter=',')

    figure_file = os.path.join(args.output_path, name + '.salience.png')
    dpi = 120
    fig = plt.figure(figsize=(salience.shape[0] / dpi, salience.shape[1] / dpi), dpi=dpi)
    fig.figimage(salience.transpose(), vmin=0, vmax=1, cmap='inferno', origin='lower')
    fig.savefig(figure_file)
    plt.close(fig)

    if args.save_numpy:
        salience_file = os.path.join(args.output_path, name + '.salience.npy')
        np.save(salience_file, salience)
