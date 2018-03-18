import os
from typing import Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat

from .. import Dataset
from ..utils import close_iterator


def tf_record_iterator(path, options=None):
    """A safer version of tf_record_iterator; using until #17750 is merged

    Args:
    path: The path to the TFRecords file.
    options: (optional) A TFRecordOptions object.

    Yields:
    Strings.

    Raises:
    IOError: If `path` cannot be opened for reading.
    """
    compression_type = tf.python_io.TFRecordOptions.get_compression_type_string(options)
    with errors.raise_exception_on_not_ok_status() as status:
        reader = pywrap_tensorflow.PyRecordReader_New(
            compat.as_bytes(path), 0, compat.as_bytes(compression_type), status)

    if reader is None:
        raise IOError("Could not open %s." % path)
    try:
        while True:
            try:
                with errors.raise_exception_on_not_ok_status() as status:
                    reader.GetNext(status)
            except errors.OutOfRangeError:
                break
            yield reader.record()
    finally:
        reader.Close()


# TODO: faster protobuf->numpy conversion
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
        iterator = tf_record_iterator(file, options)
        try:
            for record in iterator:
                example = tf.train.Example()
                example.ParseFromString(record)
                feature_map = example.features.feature
                if keys is None:
                    keys = list(feature_map.keys())
                yield {key: parse_feature(feature_map[key]) for key in keys}
        finally:
            close_iterator(iterator)


def tfrecord(*paths: str, keys: List[str]=None, compression: Optional[str]=None, **executor_config):
    options = None
    if compression and compression.lower() == 'gzip':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    elif compression and compression.lower() == 'zlib':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    return Dataset(paths).flatmap(lambda path: read_records(path, keys, options), **executor_config)
