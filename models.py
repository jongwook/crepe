from keras.layers import *
from keras.models import Model


def crepe(optimizer, model_capacity=32, **_) -> Model:
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * model_capacity for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="conv%d-maxpool" % layer)(y)
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model


def rescrepe(optimizer, model_capacity=32, **_) -> Model:
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    filters = [n * model_capacity for n in [8, 4, 4, 2, 2, 4, 4, 8]]
    widths = [512, 256, 128, 64, 64, 32, 16, 8]
    strides = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        yy = y
        if K.int_shape(yy)[-1] != filters:
            yy = Conv2D(filters, (1, 1))(yy)
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="conv%d-maxpool" % layer)(Add()([y, yy]))
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model


def short(optimizer, model_capacity=32, **_) -> Model:
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    filters = [n * model_capacity for n in [8, 8, 8, 16, 16, 16, 16, 32, 32, 32]]
    widths = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    y = Conv2D(128, (3, 1), padding='same', activation='relu', name="conv0")(y)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="conv%d-maxpool" % layer)(y)
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model


def dilated(optimizer, model_capacity=32, **_) -> Model:
    layers = 30
    filters = [model_capacity * 4] * layers
    widths = [3] * layers

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, width in zip(range(layers), filters, widths):
        dilation = 2 ** (layer % 10)
        yy = y
        y = Conv2D(filters, (width, 1), dilation_rate=(dilation, 1),
                   padding='same', activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = Conv2D(filters, (1, 1), activation='linear', name="conv%d-1x1" % layer)(y)
        y = Add()([y, yy])

    y = AvgPool2D(pool_size=(64, 1), padding='valid', name="avgpool")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model
