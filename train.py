from config import *
from datasets import *
from evaluation import raw_pitch_accuracy


def prepare_data() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset('rwcsynth', batch_size=options['batch_size'])
    print("Train dataset:", train)

    print("Collecting validation set:")
    validation = validation_dataset('mir1k', sample_files=50, take=200, seed=42).collect(verbose=True)
    print("Validation data:", validation[0].shape, validation[1].shape)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_data):
        self.audio = val_data[0]
        self.true_cents = to_weighted_average_cents(val_data[1])

    def on_epoch_end(self, epoch, _):
        predicted = self.model.predict(self.audio, verbose=1)
        predicted_cents = to_weighted_average_cents(predicted)
        mae = np.mean(np.abs(self.true_cents - predicted_cents))
        rpa = raw_pitch_accuracy(self.true_cents, predicted_cents)

        print("Epoch {}: MAE = {}, RPA = {}".format(epoch, mae, rpa))


def main():
    train_data, val_data = prepare_data()

    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_data), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [PitchAccuracyCallback(val_data)],
                        validation_data=val_data)


if __name__ == "__main__":
    main()
