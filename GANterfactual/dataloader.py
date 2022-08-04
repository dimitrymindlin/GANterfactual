from __future__ import print_function, division
import tensorflow as tf
from skimage.io import imread
import numpy as np
from keras.utils.all_utils import Sequence


class Test_img_data_generator(Sequence):
    """
    Take 15 positive self selected examples and try counterfactual generation
    """

    def __init__(self, image_filenames, labels, batch_size, img_height, img_width):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        root = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/"
        self.pos_image_paths = [root + "patient11186/study2_positive/image1.png",
                                root + "patient11186/study2_positive/image2.png",
                                root + "patient11186/study2_positive/image3.png",
                                root + "patient11186/study3_positive/image1.png",
                                root + "patient11186/study3_positive/image2.png",
                                root + "patient11186/study3_positive/image3.png",
                                root + "patient11188/study1_positive/image1.png",
                                root + "patient11188/study1_positive/image2.png",
                                root + "patient11188/study1_positive/image3.png",
                                root + "patient11188/study1_positive/image4.png",
                                root + "patient11190/study1_positive/image1.png",
                                root + "patient11190/study1_positive/image2.png",
                                root + "patient11192/study1_positive/image1.png",
                                root + "patient11192/study1_positive/image2.png",
                                root + "patient11192/study1_positive/image3.png"]

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # TODO: ONLY FOR BATCH SIZE OF 1
        batch_pos = [self.pos_image_paths[idx % len(self.pos_image_paths)]]
        batches = [batch_pos]
        pos = []
        for i, batch in enumerate(batches):
            for file in batch:
                img = imread(file)
                if len(img.shape) < 3:
                    img = tf.expand_dims(img, axis=-1)
                if img.shape[-1] != 3:
                    img = tf.image.grayscale_to_rgb(img)
                img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
                pos.append(img / 127.5 - 1.)
        pos = tf.stack(pos)
        return pos
