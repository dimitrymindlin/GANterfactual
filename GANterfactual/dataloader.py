from __future__ import print_function, division

import os
import tensorflow as tf
import numpy as np
from data.mura_dataset import MuraDataset


class DataLoader():
    def __init__(self, dataset_name=None, img_res=(128, 128), config=None):
        self.dataset_name = dataset_name
        self.config = config
        self.dataset = MuraDataset(config=config)
        self.img_res = img_res

        self.image_gen_config = {
            "horizontal_flip": False,
            "preprocessing_function": (lambda x: x / 127.5 - 1.),
            "rescale": None,
        }

    def load_batch(self):
        for pos, neg in zip(self.dataset.ds_train_pos, self.dataset.ds_train_neg):
            yield pos[0], neg[0]

    def load_single(self, path):
        img = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=self.img_res)
        x = tf.keras.preprocessing.image.img_to_array(img) / 127.5 - 1
        return x

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        tf.keras.preprocessing.image.save_img(path, x)
