from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from data.mura_dataset import MuraDataset
from tensorflow import keras
import tensorflow as tf
import cv2
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence


class DataLoader():
    def __init__(self, config=None):
        self.config = config
        #self.dataset = MuraDataset(config=config)
        self.train_dataloader, self.test_dataloader, self.clf_test_data = get_mura_data()

    def load_batch(self):
        for pos, neg in self.train_dataloader:
            # pos = class label 1, neg = class label 0
            yield neg, pos  # "NORMAL, ABNORMAL"

    def load_single(self):
        # TODO CHANGE TO METHOD that returns many samples of pos and neg
        self.dataset.ds_test.shuffle(self.dataset.ds_info.splits['test'].num_examples)
        samples = [(x, y) for x, y in self.dataset.ds_test.take(1)]
        return samples[0][0][0], samples[0][1][0]

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        tf.keras.preprocessing.image.save_img(path, x)


class Gan_data_generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, transform):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.pos_image_paths = [filename for filename in image_filenames if
                                "positive" in filename]
        self.neg_image_paths = [filename for filename in image_filenames if
                                "negative" in filename]
        self.t = transform

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_pos = self.pos_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_neg = self.neg_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batches = [batch_neg, batch_pos]
        pos = []
        neg = []
        for i, batch in enumerate(batches):
            for file in batch:
                print(file)
                img = imread(file)
                img = self.t(image=img)["image"]
                if len(img.shape) < 3:
                    img = tf.expand_dims(img, axis=-1)
                if img.shape[-1] != 3:
                    img = tf.image.grayscale_to_rgb(img)
                img = tf.image.resize_with_pad(img, 224, 224)
                if i == 1:
                    pos.append(img / 127.5 - 1.)
                else:
                    neg.append(img / 127.5 - 1.)
        neg = tf.stack(neg)
        pos = tf.stack(pos)
        return neg, pos

class CLFDataGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size, transform):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.t = transform

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        x = []
        for file in batch_x:
            img = imread(file)
            img = self.t(image=img)["image"]
            if len(img.shape) < 3:
                img = tf.expand_dims(img, axis=-1)
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.resize_with_pad(img, 224, 224)
            img = tf.cast(img, tf.float32) / 127.5 - 1.
            x.append(img)
        x = tf.stack(x)
        y = np.array(batch_y)
        return x, y

def get_mura_data():
    # To get the filenames for a task
    def filenames(part, train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        #root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    from albumentations import (
        Compose, HorizontalFlip, CLAHE, HueSaturationValue,
        RandomBrightness, RandomContrast, RandomGamma,
        ToFloat, ShiftScaleRotate
    )

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        ToFloat(max_value=255)
    ])
    AUGMENTATIONS_TEST = Compose([
        # CLAHE(p=1.0, clip_limit=2.0),
        ToFloat(max_value=255)
    ])

    part = 'XR_WRIST'  # part to work with
    imgs, labels = filenames(part=part)  # train data
    vimgs, vlabels = filenames(part=part, train=False)  # validation data

    #training_data = labels.count('positive') + labels.count('negative')
    #validation_data = vlabels.count('positive') + vlabels.count('negative')

    y_data = [0 if x == 'negative' else 1 for x in labels]
    y_data = keras.utils.to_categorical(y_data)
    y_data_valid = [0 if x == 'negative' else 1 for x in vlabels]
    y_data_valid = keras.utils.to_categorical(y_data_valid)

    batch_size = 1
    imgs, y_data = shuffle(imgs, y_data)
    training_batch_generator = Gan_data_generator(imgs, y_data, batch_size, AUGMENTATIONS_TRAIN)
    validation_batch_generator = Gan_data_generator(vimgs, y_data_valid, batch_size, AUGMENTATIONS_TEST)
    clf_test_data_generator = CLFDataGenerator(vimgs, y_data_valid, batch_size, AUGMENTATIONS_TEST)

    return training_batch_generator, validation_batch_generator, clf_test_data_generator