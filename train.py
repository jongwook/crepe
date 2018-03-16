from options import options, build_model
from datasets import *
from tensorflow.python.keras.models import Model

train = train_dataset(batch_size=8)
validation = validation_dataset('nsynth-test')

model: Model = build_model()
model.summary()

model.fit_generator(iter(train), steps_per_epoch=1000, epochs=10)
