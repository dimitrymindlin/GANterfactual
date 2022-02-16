# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf
import tensorflow_addons as tfa
from utils.model_utils import get_model_by_name, get_input_shape_from_config, get_preprocessing_by_name


class WristPredictNet(tf.keras.Model):
    """MuraNet Model Class with various base models"""

    def __init__(self, config, weights='imagenet', include_top=True):
        super(WristPredictNet, self).__init__(name='WristPredictNet')
        self.config = config
        self.include_top = include_top
        self._input_shape = get_input_shape_from_config(self.config)
        self.base_model = get_model_by_name(self.config, self._input_shape, weights)
        self.base_model.trainable = self.config['train']['train_base']
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

    def call(self, x):
        x = self.base_model(x)
        if self.include_top:
            return self.classifier(x)
        else:
            return x


class PreprocessNet(tf.keras.Model):
    """Mura data preprocessing"""

    def __init__(self, config):
        super(PreprocessNet, self).__init__(name='PreprocessNet')
        self.config = config
        self._input_shape = get_input_shape_from_config(self.config)
        self.preprocessing_layer = get_preprocessing_by_name(self.config, self._input_shape)
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)

    def call(self, inputs):
        x = tfa.image.equalize(inputs)
        x = resize_with_pad(x, self.config["data"]["image_height"], self.config["data"]["image_width"])
        x = self.preprocessing_layer(x)  # Normalisation to [0,1]
        if self.config['train']['augmentation']:
            x = self.random_flipping_aug(x)
            x = self.random_rotation_aug(x)
        return x


def resize_with_pad(image, height, width):
    return tf.image.resize_with_pad(image, height, width)


def get_mura_model(config, include_top=True):
    input_shape = get_input_shape_from_config(config)
    inputs = tf.keras.Input(shape=input_shape)
    pre = PreprocessNet(config)(inputs)
    wrist_net = WristPredictNet(config, include_top=include_top)(pre)
    return tf.keras.Model(inputs, wrist_net)


def get_finetuning_model_from_pretrained_model(model):
    x = tf.keras.layers.Dropout(0.2)(model.layers[-1].output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.layers[0].input, outputs=x)
    return model
