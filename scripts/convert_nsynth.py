import os
import tensorflow as tf
from tqdm import tqdm

search_root = os.path.join('..', 'original')
files = os.listdir(search_root)
files = [file for file in files if file.endswith('.tfrecord')]
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

for file in tqdm(files):
    input_path = os.path.join(search_root, file)
    output_path = os.path.join('..', file)
    writer = tf.python_io.TFRecordWriter(output_path, options=options)
    for record in tf.python_io.tf_record_iterator(input_path):
        example = tf.train.Example()
        example.ParseFromString(record)

        audio = example.features.feature['audio'].float_list.value
        pitch = example.features.feature['pitch'].int64_list.value[0]

        freq = 2 ** ((pitch - 69) / 12.0) * 440

        for i in range(4):
            example = tf.train.Example(features=tf.train.Features(feature={
                "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio[0:1024])),
                "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[freq]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
