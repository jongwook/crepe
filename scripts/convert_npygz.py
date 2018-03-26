import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source_dir')
parser.add_argument('target_dir')
args = parser.parse_args()

import os
import gzip
import numpy as np
import tensorflow as tf
from tqdm import tqdm

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

frequencies_files = [file for file in os.listdir(os.path.join(args.source_dir, 'frequencies')) if file.endswith('.npy.gz')]
audio_files = [file for file in os.listdir(os.path.join(args.source_dir, 'raw')) if file.endswith('.npy.gz')]

frequencies_files.sort()
audio_files.sort()

assert len(frequencies_files) == len(audio_files)
assert len(frequencies_files) > 0

for frequency_file, audio_file in tqdm(list(zip(frequencies_files, audio_files))):
    freqs = np.load(gzip.open(os.path.join(args.source_dir, 'frequencies', frequency_file)))
    audio = np.load(gzip.open(os.path.join(args.source_dir, 'raw', audio_file)))

    assert audio.shape[1] == freqs.shape[0]

    output_path = os.path.join(args.target_dir, frequency_file.replace('.npy.gz', '.tfrecord'))
    writer = tf.python_io.TFRecordWriter(output_path, options=options)

    nonzero = freqs > 0
    audio = audio[:, nonzero]
    freqs = freqs[nonzero]

    for i in tqdm(range(freqs.shape[0])):
        example = tf.train.Example(features=tf.train.Features(feature={
            "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio[:, i])),
            "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[freqs[i]]))
        }))
        writer.write(example.SerializeToString())

    writer.close()
