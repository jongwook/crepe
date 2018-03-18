from config import *
from datasets import *

import tensorflow as tf


def prepare_data() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(batch_size=options['batch_size'])

    print("Train dataset:", train)
    print("Collecting validation set:")

    validation = validation_dataset('mir1k', sample_files=100, take=100, seed=42).collect(verbose=True)
    print("Validation data:", validation[0].shape, validation[1].shape)

    return train, validation


def get_callbacks() -> List[callable]:
    cb = tf.keras.callbacks
    result: List[cb.Callback] = [
        cb.CSVLogger(log_path('learning-curve.tsv'), separator='\t'),
    ]

    if options['save_model_weights']:
        result.append(cb.ModelCheckpoint(log_path(options['save_model_weights']), save_weights_only=True))
    elif options['save_model']:
        result.append(cb.ModelCheckpoint(log_path(options['save_model'])))

    if options['tensorboard']:
        result.append(cb.TensorBoard(log_path('tensorboard')))

    return result


if __name__ == "__main__":
    train, validation = prepare_data()

    model: tf.keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_callbacks(), validation_data=validation)
