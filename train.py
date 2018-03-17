from tensorflow.python.keras.models import Model

from config import options, build_model, callbacks
from datasets import train_dataset, validation_dataset

train = train_dataset(batch_size=options['batch_size'])
validation = validation_dataset('nsynth-test')

model: Model = build_model()
model.summary()

model.fit_generator(iter(train), steps_per_epoch=options['steps_per_epoch'], epochs=options['epochs'],
                    callbacks=callbacks)
