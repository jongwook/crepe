import os
import tensorflow as tf
import librosa
from tqdm import tqdm

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

directories = [[d for d in os.listdir('original') if d.startswith("%02d-" % i)][0] for i in range(1, 11)]
pitch_files = [[f for f in os.listdir('original') if f.startswith("%02d_" % i)][0] for i in range(1, 11)]

suffixes = ['-violin.wav', '-clarinet.wav', '-saxphone.wav', '-bassoon.wav']

for label_file, directory in tqdm(list(zip(pitch_files, directories))):
    labels = open(os.path.join('original', label_file)).read().split("\n")
    labels = [line.split("\t") for line in labels]
    labels = [[float(n) for n in row] for row in labels if len(row) == 5]

    sr = 16000
    audios = [librosa.load(os.path.join('original', directory, directory + suffixes[i]), sr=sr)[0] for i in range(4)]

    output_path = directory + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(output_path, options=options)

    for row in tqdm(labels):
        center = int(row[0] * sr)
        pitches = row[1:]
        for i in range(4):
            segment = audios[i][center-512:center+512]
            example = tf.train.Example(features=tf.train.Features(feature={
                "audio": tf.train.Feature(float_list=tf.train.FloatList(value=segment)),
                "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[pitches[i]]))
            }))
            writer.write(example.SerializeToString())

    writer.close()