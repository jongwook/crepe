from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model, Sequential


def linear() -> Model:
    return Sequential([
        Dense(1, input_shape=(1024, 1))
    ])
