import sys
from config import *
from datasets import *
from evaluation import accuracies


validation_set_names = ['medleydb', 'rwcsynth', 'nsynth-test', 'nsynth-valid']


def prepare_datasets() -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(batch_size=options['batch_size'], augment=options['augment'])
    print("Train dataset:", train, file=sys.stderr)

    validation = []
    for name in validation_set_names:
        print("Collecting validation set {}:".format(name), file=sys.stderr)
        dataset = validation_dataset(name, seed=42, take=100).take(options['validation_take']).collect(verbose=True)
        validation.append(dataset)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_sets):
        self.val_sets = [(audio, to_weighted_average_cents(pitch)) for audio, pitch in val_sets]
        for filename in ["eval-mae.tsv", "eval-rpa.tsv", "eval-rca.tsv"]:
            with open(log_path(filename), "w") as f:
                f.write('\t'.join(validation_set_names) + '\n')

    def on_epoch_end(self, epoch, _):
        names = list(validation_set_names)
        print(file=sys.stderr)

        MAEs = []
        RPAs = []
        RCAs = []

        for audio, true_cents in self.val_sets:
            predicted = self.model.predict(audio)
            predicted_cents = to_weighted_average_cents(predicted)
            diff = np.abs(true_cents - predicted_cents)
            mae = np.mean(diff[np.isfinite(diff)])
            rpa, rca = accuracies(true_cents, predicted_cents)
            nans = np.mean(np.isnan(diff))
            print("Epoch {}, Dataset {}: MAE = {}, RPA = {}, RCA = {}, nans = {}".format(epoch, names.pop(0), mae, rpa, rca, nans), file=sys.stderr)
            MAEs.append(mae)
            RPAs.append(rpa)
            RCAs.append(rca)

        with open(log_path("eval-mae.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % mae for mae in MAEs]) + '\n')
        with open(log_path("eval-rpa.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in RPAs]) + '\n')
        with open(log_path("eval-rca.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rca for rca in RCAs]) + '\n')

        print(file=sys.stderr)


def main():
    train_set, val_sets = prepare_datasets()
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()

    model: keras.Model = build_model()
    model.summary()

    model.fit_generator(iter(train_set), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                        callbacks=get_default_callbacks() + [PitchAccuracyCallback(val_sets)],
                        validation_data=val_data)


if __name__ == "__main__":
    K = tf.keras.backend

    main()
