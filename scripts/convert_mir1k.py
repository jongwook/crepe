import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

label_files = [file for file in os.listdir('PitchLabel') if file.endswith('.pv')]
audio_files = [file for file in os.listdir('WavFile') if file.endswith('.wav')]

label_files.sort()
audio_files.sort()

assert len(label_files) == len(audio_files)

print(len(audio_files), 'files found')

for label_file, audio_file in tqdm(list(zip(label_files, audio_files))):
    labels = np.loadtxt(os.path.join('PitchLabel', label_file))
    audio, sr = librosa.load(os.path.join('WavFile', audio_file), sr=16000)

    labels = labels[8:-8, :]  # trim the ends
    nonzero = labels[:, 1] > 0
    labels = labels[nonzero, :]

    output_path = audio_file.replace('.wav', '.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path, options=options)

    for i in range(labels.shape[0]):
        center = int(labels[i, 0] * sr)
        segment = audio[center-512:center+512]
        freq = labels[i, 1]
        example = tf.train.Example(features=tf.train.Features(feature={
            "audio": tf.train.Feature(float_list=tf.train.FloatList(value=segment)),
            "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[freq]))
        }))
        writer.write(example.SerializeToString())

    writer.close()
