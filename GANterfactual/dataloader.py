from __future__ import print_function, division

import os
import tensorflow as tf
import numpy as np
from data.mura_dataset import MuraDataset


class DataLoader():
    def __init__(self, img_res=(128, 128), config=None):
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

    def load_single(self):
        self.dataset.ds_test.shuffle(self.dataset.ds_info.splits['test'].num_examples)
        samples = [(x,y) for x, y in self.dataset.ds_test.take(1)]
        index = np.random.randint(0,8)
        return samples[0][0][index], samples[0][1][index]

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        tf.keras.preprocessing.image.save_img(path, x)
