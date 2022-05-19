from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence


class DataLoader():
    def __init__(self, config=None):
        self.config = config
        self.img_height = config["data"]["image_height"]
        self.img_width = config["data"]["image_width"]
        self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.clf_test_data, self.test_y = get_mura_data(
            self.img_height, self.img_width)

    def load_batch(self):
        max_iterations = max(len(self.train_dataloader.pos_image_paths), len(self.train_dataloader.neg_image_paths))
        for (neg, pos), i in zip(self.train_dataloader, range(max_iterations)):
            # neg = class label 0 = normal, pos = class label 1 = abnormal
            yield neg, pos  # "NORMAL, ABNORMAL"

    def load_test(self):
        for neg, pos in self.test_dataloader:
            yield neg, pos

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        tf.keras.preprocessing.image.save_img(path, x)


class Gan_data_generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, img_height, img_width):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.pos_image_paths = [filename for filename in image_filenames if
                                "positive" in filename]
        self.neg_image_paths = [filename for filename in image_filenames if
                                "negative" in filename]
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # TODO: ONLY FOR BATCH SIZE OF 1
        if idx > len(self.neg_image_paths):
            batch_neg = None
        else:
            batch_neg = [self.neg_image_paths[idx % len(self.neg_image_paths)]]
        if idx > len(self.pos_image_paths):
            batch_pos = None
        else:
            batch_pos = [self.pos_image_paths[idx % len(self.pos_image_paths)]]
        batches = [batch_neg, batch_pos]
        pos = []
        neg = []
        for i, batch in enumerate(batches):
            if batch != None:
                for file in batch:
                    img = imread(file)
                    if len(img.shape) < 3:
                        img = tf.expand_dims(img, axis=-1)
                    if img.shape[-1] != 3:
                        img = tf.image.grayscale_to_rgb(img)
                    img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
                    if i == 0:
                        neg.append(img / 127.5 - 1.)
                    else:
                        pos.append(img / 127.5 - 1.)
        neg = tf.stack(neg)
        pos = tf.stack(pos)
        return neg, pos


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


class CLFDataGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size, img_height=None, img_width=None):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = 32
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        x = []
        for file in batch_x:
            img = imread(file)
            if len(img.shape) < 3:
                img = tf.expand_dims(img, axis=-1)
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
            img = tf.cast(img, tf.float32) / 127.5 - 1.
            x.append(img)
        x = tf.stack(x)
        y = np.array(batch_y)
        return x, y


def get_mura_data(img_height, img_width):
    # To get the filenames for a task
    def filenames(part, train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

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

    part = 'XR_WRIST'  # part to work with
    train_x, train_y = filenames(part=part)  # train data
    test_x_filenames, test_y = filenames(part=part, train=False)  # test data
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2,
                                                          random_state=42)  # split train and valid data

    train_x, train_y = to_categorical(train_x, train_y)
    valid_x, valid_y = to_categorical(valid_x, valid_y)
    test_x_filenames, test_y = to_categorical(test_x_filenames, test_y)

    batch_size = 1
    train_batch_generator = Gan_data_generator(train_x, train_y, batch_size, img_height, img_width)
    valid_batch_generator = Gan_data_generator(valid_x, valid_y, batch_size, img_height, img_width)
    test_batch_generator = Gan_data_generator(test_x_filenames, test_y, batch_size, img_height, img_width)
    clf_test_data_generator = CLFDataGenerator(test_x_filenames, test_y, batch_size, img_height, img_width)

    return train_batch_generator, valid_batch_generator, test_batch_generator, clf_test_data_generator, test_y


def to_categorical(x, y):
    y = [0 if x == 'negative' else 1 for x in y]
    y = keras.utils.to_categorical(y)
    x, y = shuffle(x, y)
    return x, y
