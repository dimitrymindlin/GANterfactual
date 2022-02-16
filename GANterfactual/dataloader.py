from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from data.mura_dataset import MuraDataset


class DataLoader():
    def __init__(self, config=None):
        self.config = config
        self.dataset = MuraDataset(config=config)


    def load_batch(self):
        for pos, neg in zip(self.dataset.ds_train_pos, self.dataset.ds_train_neg):
            # pos = class label 1, neg = class label 0
            yield neg[0], pos[0]  # "NORMAL, ABNORMAL"

    def load_single(self):
        self.dataset.ds_test.shuffle(self.dataset.ds_info.splits['test'].num_examples)
        samples = [(x, y) for x, y in self.dataset.ds_test.take(1)]
        return samples[0][0][0], samples[0][1][0]

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        tf.keras.preprocessing.image.save_img(path, x)
