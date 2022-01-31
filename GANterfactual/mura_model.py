# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf


class WristPredictNet(tf.keras.Model):
    """MuraDenseNet Model Class with various base models"""

    def __init__(self, config, weights='imagenet', train_base=False):
        super(WristPredictNet, self).__init__(name='WristPredictNet')
        self.config = config
        self._input_shape = (
            config['data']['image_height'],
            config['data']['image_width'],
            config['data']['image_channel']
        )
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.preprocessing_layer = tf.keras.applications.densenet.preprocess_input
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
        self.base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                       input_tensor=self.img_input,
                                                       input_shape=self._input_shape,
                                                       weights=weights,
                                                       pooling=config['model']['pooling'],
                                                       classes=len(config['data']['class_names']))
        self.base_model.trainable = train_base
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

    def call(self, x):
        x = self.preprocessing_layer(x)
        if self.config['train']['augmentation']:
            x = self.random_flipping_aug(x)
            x = self.random_rotation_aug(x)
        x = self.base_model(x)
        return self.classifier(x)
