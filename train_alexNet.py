from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import ImageFile

from GANterfactual.classifier import get_adapted_alexNet
from GANterfactual.domain_to_domain_model import Domain2DomainModel

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(1000)
batch_size = 32
image_size = 512
if len(tf.config.list_physical_devices('GPU')) == 0:
    TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets/rsna_data"
else:
    TFDS_PATH = "../tensorflow_datasets/rsna_data"
TF_LOG_DIR = 'tensorboard_logs/alexNet/' + TIMESTAMP + "/"
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
        color_mode='grayscale')

    validation_data = train_gen.flow_from_directory(
        directory=f"{TFDS_PATH}/validation",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale')

    test_data = train_gen.flow_from_directory(
        directory=f"{TFDS_PATH}/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale')

    return train_data, validation_data, test_data


model = get_adapted_alexNet(image_size)
"""model = Domain2DomainModel(img_shape=(image_size, image_size, 3)).model()
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=2, from_logits=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc])"""
# model.summary()

train, validation, test = get_data(batch_size=batch_size)

weights_path = f"checkpoints/alexNet/alexNet_{TIMESTAMP}.h5"
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                    # Callback to save the Keras model or model weights at some frequency.
                                    monitor='val_accuracy',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto',
                                    save_freq='epoch'),
    keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                histogram_freq=1,
                                write_graph=True,
                                write_images=False,
                                update_freq='epoch',
                                profile_batch=30,
                                embeddings_freq=0,
                                embeddings_metadata=None
                                ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=4,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                                  )
]

if __name__ == "__main__":
    hist = model.fit(train,
                     verbose=1,
                     class_weight=None,
                     epochs=1000,
                     validation_data=validation,
                     callbacks=my_callbacks)

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
