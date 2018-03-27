import sys
from config import *
from datasets import *
from evaluation import accuracies


def prepare_datasets() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(batch_size=options['batch_size'])
    print("Train dataset:", train, file=sys.stderr)

    validation = []
    for name in ['medleydb', 'rwcsynth', 'nsynth-test', 'nsynth-valid']:
        print("Collecting validation set {}:".format(name), file=sys.stderr)
        dataset = validation_dataset(name, seed=42, take=100).take(2000).collect(verbose=True)
        validation.append(dataset)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_sets):
        self.val_sets = [(audio, to_weighted_average_cents(pitch)) for audio, pitch in val_sets]

    def on_epoch_end(self, epoch, _):
        names = ['medleydb', 'rwcsynth', 'nsynth-test', 'nsynth-valid']
        for audio, true_cents in self.val_sets:
            predicted = self.model.predict(audio)
            predicted_cents = to_weighted_average_cents(predicted)
            diff = np.abs(true_cents - predicted_cents)
            mae = np.mean(diff[np.isfinite(diff)])
            rpa, rca = accuracies(true_cents, predicted_cents)
            print("Epoch {}, Dataset {}: MAE = {}, RPA = {}, RCA = {}".format(epoch, names.pop(0), mae, rpa, rca), file=sys.stderr)
        print(file=sys.stderr)


def main():
    train_set, val_sets = prepare_datasets()

    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_set), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [PitchAccuracyCallback(val_sets)])


if __name__ == "__main__":
    K = tf.keras.backend

    main()
