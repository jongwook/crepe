import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source_dir')
parser.add_argument('target_dir')
parser.add_argument('num_splits', type=int)
parser.add_argument('--prefix', '-p', default='data')
args = parser.parse_args()

source_dir = args.source_dir
target_dir = args.target_dir
num_splits = args.num_splits
prefix = args.prefix

import os

assert os.path.isdir(source_dir)
assert os.path.isdir(target_dir)
assert num_splits > 0

import hashlib
import random
from tqdm import tqdm
import tensorflow as tf

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
output_files = [os.path.join(target_dir, '%s-%04d.tfrecord' % (prefix, i)) for i in range(num_splits)]
writers = [tf.python_io.TFRecordWriter(file, options=options) for file in output_files]

files = [file for file in os.listdir(source_dir) if file.endswith('.tfrecord')]

# distribute to splits
for file in tqdm(files):
    for record in tf.python_io.tf_record_iterator(os.path.join(source_dir, file), options=options):
        split = int(hashlib.sha256(record).hexdigest()[-8:], 16) % num_splits
        writers[split].write(record)

for writer in writers:
    writer.close()

# shuffle per-file
for file in tqdm(output_files):
    records = [record for record in tf.python_io.tf_record_iterator(file, options=options)]
    random.shuffle(records)
    writer = tf.python_io.TFRecordWriter(file, options=options)
    for record in records:
        writer.write(record)
    writer.close()
