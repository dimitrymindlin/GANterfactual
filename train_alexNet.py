from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization
import numpy as np

import os

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from PIL import ImageFile
from tensorflow.python.keras.regularizers import l2

TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(1000)
dimension = 512
dataset_path = "../tensorflow_datasets/rsna_data"
TF_LOG_DIR = 'tensorboard_logs/alexNet/' + TIMESTAMP + "/"


def get_adapted_alexNet():
    input = Input(shape=(dimension, dimension, 1))

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


def get_data():
    image_size = dimension
    batch_size = 32
    # Load rsna_data for training
    train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=(lambda x: x / 127.5 - 1.))

    train_data = train_gen.flow_from_directory(
        directory=f"{dataset_path}/train",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='grayscale')

    validation_data = train_gen.flow_from_directory(
        directory=f"{dataset_path}/validation",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='grayscale')

    test_data = train_gen.flow_from_directory(
        directory=f"{dataset_path}/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='grayscale')

    return train_data, validation_data, test_data


model = get_adapted_alexNet()
# model.summary()

train, validation, test = get_data()
weights_path = f"checkpoints/alexNet/alexNet_{TIMESTAMP}.h5"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                    histogram_freq=1,
                                    write_graph=True,
                                    write_images=False,
                                    update_freq='epoch',
                                    profile_batch=30,
                                    embeddings_freq=0,
                                    embeddings_metadata=None
                                    )
check_point = keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True, monitor='val_accuracy', mode='max',
                                              save_weights_only=True)
early_stopping = keras.callbacks.EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

if __name__ == "__main__":
    hist = model.fit_generator(train,
                               epochs=1000,
                               validation_data=validation,
                               callbacks=[check_point, early_stopping, tensorboard_callback],
                               steps_per_epoch=len(train),
                               validation_steps=len(validation))

    print("Training done, best weights saved. Trying to save whole model:")
    # model.save(os.path.join('', 'models', 'classifier', 'model.h5'), include_optimizer=False)
    model.load_weights(weights_path)
    print("Loaded weights")
    tf.keras.models.save_model(model, os.path.join('', 'models', 'classifier', f'alexNet_{TIMESTAMP}_model.h5'),
                               include_optimizer=True, )
    print("Train History")
    print(hist)
    print("Evaluation")
    result = model.evaluate(test)
    result = dict(zip(model.metrics_names, result))
    result_matrix = [[k, str(w)] for k, w in result.items()]
    for metric, value in zip(model.metrics_names, result):
        print(metric, ": ", value)
