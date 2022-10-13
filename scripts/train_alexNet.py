from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import ImageFile

from GANterfactual.classifier import get_adapted_alexNet

TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
MODEL = "alexNet_rsna"
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(1000)
batch_size = 32
image_size = 512
if len(tf.config.list_physical_devices('GPU')) == 0:
    TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets/rsna_data"
else:
    TFDS_PATH = "../tensorflow_datasets/rsna_data"
TF_LOG_DIR = f'tensorboard_logs/{MODEL}/' + TIMESTAMP + "/"
ckp_path = f"checkpoints/{MODEL}/{MODEL}_{TIMESTAMP}/"
file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
with file_writer.as_default():
    tf.summary.text("TS", TIMESTAMP, step=0)


def plot_any_img(img):
    if np.min(img) < 0:
        img = tf.math.add(tf.math.multiply(0.5, img), 0.5)
    plt.imshow(np.squeeze(img), vmin=np.min(img), vmax=np.max(img), cmap=plt.get_cmap('gray'))
    plt.show()


def get_data(batch_size):
    # Load rsna_data for training
    train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=(lambda x: x / 127.5 - 1.))

    train_data = train_gen.flow_from_directory(
        directory=f"{TFDS_PATH}/train",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='grayscale',
        classes={'normal': 0,
                 'abnormal': 1}
    )

    validation_data = train_gen.flow_from_directory(
        directory=f"{TFDS_PATH}/validation",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale',
        classes={'normal': 0,
                 'abnormal': 1}
    )

    test_data = train_gen.flow_from_directory(
        directory=f"{TFDS_PATH}/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale',
        classes={'normal': 0,
                 'abnormal': 1}
    )

    return train_data, validation_data, test_data


if __name__ == "__main__":
    model = get_adapted_alexNet(image_size)
    train, validation, test = get_data(batch_size=batch_size)
    reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                          factor=0.1,
                                                          patience=3,
                                                          min_delta=0.001,
                                                          verbose=1,
                                                          min_lr=1e-8),
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=False,
                                                       update_freq='epoch',
                                                       profile_batch=30,
                                                       embeddings_freq=0,
                                                       embeddings_metadata=None
                                                       )
    check_point = keras.callbacks.ModelCheckpoint(ckp_path, save_best_only=True, monitor='val_accuracy', mode='max',
                                                  save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")

    hist = model.fit(train,
                     epochs=100,
                     validation_data=validation,
                     callbacks=[check_point, early_stopping, tensorboard_callback, reduce_on_plateau],
                     verbose=1,
                     class_weight=None
                     )

    print("Train History")
    print(hist)
    print("Evaluation")
    result = model.evaluate(test)
    result = dict(zip(model.metrics_names, result))
    for metric, value in result.items():
        print(metric, ": ", value)
    model.save(ckp_path + 'model')
