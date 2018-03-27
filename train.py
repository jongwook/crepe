import sys
from config import *
from datasets import *
from evaluation import accuracies


def prepare_data() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(batch_size=options['batch_size'])
    print("Train dataset:", train, file=sys.stderr)

    print("Collecting validation set:", file=sys.stderr)
    validation = validation_dataset(seed=42, take=100).take(4000).collect(verbose=True)
    print("Validation data:", validation[0].shape, validation[1].shape, file=sys.stderr)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_data):
        self.audio = val_data[0]
        self.true_cents = to_weighted_average_cents(val_data[1])

    def on_epoch_end(self, epoch, _):
        predicted = self.model.predict(self.audio, verbose=1)
        predicted_cents = to_weighted_average_cents(predicted)
        diff = np.abs(self.true_cents - predicted_cents)
        mae = np.mean(diff[np.isfinite(diff)])
        rpa, rca = accuracies(self.true_cents, predicted_cents)
        print("\nEpoch {}: MAE = {}, RPA = {}, RCA = {}\n".format(epoch, mae, rpa, rca), file=sys.stderr)


def main():
    train_data, val_data = prepare_data()

    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_data), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [PitchAccuracyCallback(val_data)],
                        validation_data=val_data)


if __name__ == "__main__":
    K = tf.keras.backend

    main()
