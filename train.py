from config import *
from datasets import *


def prepare_data() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(batch_size=options['batch_size'])

    print("Train dataset:", train)
    print("Collecting validation set:")

    validation = validation_dataset('mir1k', sample_files=100, take=100, seed=42).collect(verbose=True)
    print("Validation data:", validation[0].shape, validation[1].shape)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_data):
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        print(self.val_data[0].shape, self.val_data[1].shape)


def main():
    train_data, val_data = prepare_data()

    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_data), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [PitchAccuracyCallback(val_data)],
                        validation_data=val_data)


if __name__ == "__main__":
    main()
