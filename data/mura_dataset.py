import tensorflow as tf
import tensorflow_datasets as tfds

from data.mura_tfds import MuraImages


class MuraDataset():

    def __init__(self, config):
        self.config = config
        (train, validation, test), info = tfds.load(
            'MuraImages',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=self.config["dataset"]["download"],
            with_info=True,
        )
        self.ds_info = info
        self.ds_train = self._build_train_pipeline(train)
        self.ds_train_pos = self._build_pos_train_pipeline(train)
        self.ds_train_neg = self._build_neg_train_pipeline(train)
        self.ds_val = self._build_test_pipeline(validation)
        self.ds_test = self._build_test_pipeline(test)

    def _build_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_pos_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.filter(lambda x, y: y == 1)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_neg_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.filter(lambda x, y: y == 0)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds):
        ds = ds.map(
            self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config['test']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def preprocess(self, image, label):
        height = self.config['data']['image_height']
        width = self.config['data']['image_width']
        image = tf.image.resize_with_pad(image, height, width)
        return tf.cast(image, tf.float32) / 127.5 - 1., label  # normalize pixel values between -1 and 1

    def benchmark(self):
        tfds.benchmark(self.ds_train, batch_size=self.config['train']['batch_size'])
