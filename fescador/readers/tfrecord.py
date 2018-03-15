from .. import Dataset
from typing import Optional, List
import tensorflow as tf
import numpy as np
import os


def parse_feature(feature: tf.train.Feature):
    kind = feature.WhichOneof('kind')
    if kind == 'float_list':
        return np.array(feature.float_list.value, dtype=np.float32)
    if kind == 'bytes_list':
        return list(feature.bytes_list.value)
    if kind == 'int64_list':
        return np.array(feature.int64_list.value, dtype=np.int64)
    raise ValueError('unsupported feature type: {}'.format(kind))


def read_records(path: str, keys: List[str], options: tf.python_io.TFRecordOptions):
    if os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tfrecord')]
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ValueError('Could not open {}'.format(path))

    for file in files:
        for record in tf.python_io.tf_record_iterator(file, options):
            example = tf.train.Example()
            example.ParseFromString(record)
            feature_map = example.features.feature
            if keys is None:
                keys = list(feature_map.keys())
            yield {key: parse_feature(feature_map[key]) for key in keys}


def tfrecord(*paths: str, keys: List[str]=None, compression: Optional[str]=None, **executor_config):
    options = None
    if compression and compression.lower() == 'gzip':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    elif compression and compression.lower() == 'zlib':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    return Dataset(paths).flatmap(lambda path: read_records(path, keys, options), **executor_config)
