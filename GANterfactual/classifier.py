from __future__ import print_function, division
import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from PIL import ImageFile
from tensorflow.python.keras.regularizers import l2

# The trained classifier is loaded.
# Rewrite this function if you want to use another model architecture than our modified AlexNET.
# A model, which provides a 'predict' function, has to be returned.
def load_classifier(path, img_shape):
    original = keras.models.load(path)
    classifier = build_classifier(img_shape)

    counter = 0
    for layer in original.layers:
        assert (counter < len(classifier.layers))
        classifier.layers[counter].set_weights(layer.get_weights())
        counter += 1

    classifier.summary()

    return classifier


def build_classifier(img_shape):
    input = Input(shape=img_shape)

    # 1st Convolutional Layer
    x = Conv2D(filters=96,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid')(input)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation before passing it to the next layer
    x = BatchNormalization()(x, training=False)

    # 2nd Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(11, 11),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 3rd Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 4th Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 5th Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # Passing it to a dense layer
    x = Flatten()(x)
    # 1st Dense Layer
    x = Dense(4096, input_shape=img_shape)(x)
    x = Activation('relu')(x)
    # Add Dropout to prevent overfitting
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 2nd Dense Layer
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)

    # 3rd Dense Layer
    x = Dense(1000)(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x, training=False)
    # Batch Normalisation
    x = BatchNormalization()(x, training=False)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    return Model(input, x)

def get_adapted_alexNet(dimension):
    input = Input(shape=(dimension, dimension, 3))

    # 1st Convolutional Layer
    x = Conv2D(filters=96,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(input)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation before passing it to the next layer
    x = BatchNormalization()(x)

    # 2nd Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(11, 11),
               strides=(1, 1),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 3rd Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 4th Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 5th Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid')(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # Passing it to a dense layer
    x = Flatten()(x)
    # 1st Dense Layer
    x = Dense(4096,
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout to prevent overfitting
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 2nd Dense Layer
    x = Dense(4096, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 3rd Dense Layer
    x = Dense(1000, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    opt = SGD(0.0001, 0.9)
    model = Model(input, x)
    model.compile(loss='mse',
                  metrics=['accuracy'],
                  optimizer=opt)
    return model