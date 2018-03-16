from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


def crepe(optimizer, **_) -> Model:
    layers = [1, 2, 3, 4, 5, 6]
    filters = [1024, 128, 128, 128, 256, 512]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1))(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = BatchNormalization(name="conv%d-BN" % layer, epsilon=1e-5)(y)
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="conv%d" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="conv%d-maxpool" % layer)(y)
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model