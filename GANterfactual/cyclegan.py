from datetime import datetime

from tensorflow.keras import Input
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from GANterfactual.dataloader import DataLoader
from GANterfactual.discriminator import build_discriminator
from GANterfactual.generator import build_generator, ResnetGenerator
import tensorflow as tf
import os
import numpy as np
from GANterfactual.load_clf import load_classifier, load_classifier_complete
from configs.mura_pretraining_config import mura_config

execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
writer = tf.summary.create_file_writer(f'logs/' + execution_id)


class CycleGAN():
    def __init__(self, gan_config):
        self.img_rows = gan_config["data"]["image_height"]
        self.img_cols = gan_config["data"]["image_width"]
        self.channels = gan_config["data"]["image_channel"]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.gan_config = gan_config
        self.gan_config['train']['execution_id'] = execution_id
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        self.data_loader = DataLoader(config=mura_config)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = self.gan_config["train"]["cycle_consistency_loss_weight"]
        self.lambda_id = self.gan_config["train"]["identity_loss_weight"]
        self.lambda_counterfactual = self.gan_config["train"]["counterfactual_loss_weight"]

        self.d_N = None
        self.d_P = None
        self.g_NP = None
        self.g_PN = None
        self.combined = None
        self.classifier = None

    def construct(self):
        # Build the discriminators
        self.d_N = build_discriminator(self.img_shape, self.df)
        self.d_P = build_discriminator(self.img_shape, self.df)

        # Build the generators
        if self.gan_config["train"]["generator"] == "unet":
            self.g_NP = build_generator(self.img_shape, self.gf, self.channels, self.gan_config['train']['leaky_relu'])
            self.g_PN = build_generator(self.img_shape, self.gf, self.channels, self.gan_config['train']['leaky_relu'])
        else:
            self.g_NP = ResnetGenerator(self.img_shape, self.channels, self.gf)
            self.g_PN = ResnetGenerator(self.img_shape, self.channels, self.gf)

        self.build_combined()

    def load_existing(self, cyclegan_folder, load_clf=True):
        custom_objects = {"InstanceNormalization": InstanceNormalization}

        # Load discriminators from disk
        self.d_N = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_n.h5'),
                                              custom_objects=custom_objects)
        self.d_N._name = "d_N"
        self.d_P = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_p.h5'),
                                              custom_objects=custom_objects)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                               custom_objects=custom_objects)
        self.g_NP._name = "g_NP"
        self.g_PN = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                               custom_objects=custom_objects)
        self.g_PN._name = "g_PN"

        self.build_combined()

    def save(self, cyclegan_folder):
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d_N.save(os.path.join(cyclegan_folder, 'discriminator_n.h5'))
        self.d_P.save(os.path.join(cyclegan_folder, 'discriminator_p.h5'))

        # Save generators to disk
        self.g_NP.save(os.path.join(cyclegan_folder, 'generator_np.h5'))
        self.g_PN.save(os.path.join(cyclegan_folder, 'generator_pn.h5'))

    def evaluate_clf(self):
        """self.classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])"""
        print("Evaluating clf...")
        result = self.classifier.evaluate(self.data_loader.clf_test_data)
        for metric, value in zip(self.classifier.metrics_names, result):
            print(metric, ": ", value)

    def build_combined(self):
        optimizer = Adam(self.gan_config["train"]["learn_rate"],
                         self.gan_config["train"]["beta1"])

        self.d_N.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_P.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Input images from both domains
        img_N = Input(shape=self.img_shape)
        img_P = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)

        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)

        # For the combined model we will only train the generators
        self.d_N.trainable = False
        self.d_P.trainable = False

        # Discriminators determines validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        # Counterfactual loss
        #self.classifier = load_classifier(self.gan_config)
        self.classifier = load_classifier_complete(self.gan_config)
        self.classifier._name = "classifier"
        self.classifier.trainable = False
        counter_loss_N = self.classifier(fake_N)
        counter_loss_P = self.classifier(fake_P)

        # Cycle-loss - Translate images back to original domain
        cycle_N = self.g_PN(fake_P)
        cycle_P = self.g_NP(fake_N)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_N, img_P],
                              outputs=[valid_N, valid_P,
                                       counter_loss_N, counter_loss_P,
                                       cycle_N, cycle_P,
                                       img_N_id, img_P_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_counterfactual, self.lambda_counterfactual,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    def train(self, print_interval=500, sample_interval=2000):
        config_matrix = [[k, str(w)] for k, w in self.gan_config["train"].items()]
        with writer.as_default():
            tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

        batch_size = self.gan_config["train"]["batch_size"]
        epochs = self.gan_config["train"]["epochs"]

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        class_N = np.stack([np.ones(batch_size), np.zeros(batch_size)]).T
        class_P = np.stack([np.zeros(batch_size), np.ones(batch_size)]).T

        for epoch in range(epochs):
            # Positive (abnormal) = class label 1, Negative (normal) = class label 0
            for batch_i, (imgs_N, imgs_P) in enumerate(self.data_loader.load_batch()):
                # ----------------------
                #  Train Discriminators every second batch
                # ----------------------

                # Translate images to opposite domain
                # Positive (abnormal) = class label 1, Negative (normal) = class label 0
                if batch_i % self.gan_config["train"]["generator_training_multiplier"] == 0:
                    fake_P = self.g_NP.predict(imgs_N)
                    fake_N = self.g_PN.predict(imgs_P)
                    # Train the discriminators (original images = real (valid) / translated = Fake)
                    dN_loss_real = self.d_N.train_on_batch(imgs_N, valid)
                    dN_loss_fake = self.d_N.train_on_batch(fake_N, fake)
                    dN_loss = 0.5 * np.add(dN_loss_real, dN_loss_fake)

                    dP_loss_real = self.d_P.train_on_batch(imgs_P, valid)
                    dP_loss_fake = self.d_P.train_on_batch(fake_P, fake)
                    dP_loss = 0.5 * np.add(dP_loss_real, dP_loss_fake)

                    # Total disciminator loss
                    d_loss = 0.5 * np.add(dN_loss, dP_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                      [valid, valid,
                                                       class_N, class_P,
                                                       imgs_N, imgs_P,
                                                       imgs_N, imgs_P])

                # Tensorboard logging
                if self.classifier is not None:
                    if batch_i % print_interval == 0:
                        with writer.as_default():
                            tf.summary.scalar('D_loss', tf.reduce_sum(d_loss[0]), step=epoch)
                            # the accuracy of a counterfactual image generator is the percentage of
                            # counterfactuals that actually changed the classifier’s prediction
                            tf.summary.scalar('D_acc', tf.reduce_sum(100 * d_loss[1]), step=epoch)
                            tf.summary.scalar('G_loss', tf.reduce_sum(g_loss[0]), step=epoch)
                            # cycle consistency loss
                            tf.summary.scalar('Cycle_consistency', tf.reduce_sum(np.mean(g_loss[1:3])), step=epoch)
                            tf.summary.scalar('classifier_N', tf.reduce_sum(g_loss[3]), step=epoch)
                            tf.summary.scalar('classifier_P', tf.reduce_sum(g_loss[4]), step=epoch)
                            tf.summary.scalar('recon', tf.reduce_sum(np.mean(g_loss[5:7])), step=epoch)
                            tf.summary.scalar('id', tf.reduce_sum(g_loss[7:9]), step=epoch)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_N[0], imgs_P[0])

            # Comment this in if you want to save checkpoints:
            self.save(os.path.join('models', f'GANterfactual_{execution_id}', 'ep_' + str(epoch)))

    def sample_images(self, epoch, batch_i, testN, testP):
        img_folder = f"images_{execution_id}"
        os.makedirs(img_folder, exist_ok=True)
        r, c = 2, 3

        img_N = testN[np.newaxis, :, :, :]
        img_P = testP[np.newaxis, :, :, :]

        # Translate images to the other domain
        fake_P = self.g_NP.predict(img_N)
        fake_N = self.g_PN.predict(img_P)
        # Translate back to original domain
        reconstr_N = self.g_PN.predict(fake_P)
        reconstr_P = self.g_NP.predict(fake_N)

        imgs = [img_N, fake_P, reconstr_N, img_P, fake_N, reconstr_P]
        classification = [['NEGATIVE', 'POSITIVE'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

        gen_imgs = np.concatenate(imgs)
        correct_classification = ['NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE']

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(15, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{img_folder}/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def predict(self, save_pics=False):
        data_loader = DataLoader(config=mura_config)

        image_list = []
        for i in range(5):
            original, original_class = data_loader.load_single()
            x = original[np.newaxis, :, :, :]

            if original_class == 0:
                print("PREDICTION -- NEGATIVE")
                translated = self.g_NP.predict(x)
                reconstructed = self.g_PN.predict(translated)
            else:
                print("PREDICTION -- POSITIVE")
                translated = self.g_PN.predict(x)
                reconstructed = self.g_NP.predict(translated)

            imgs = [x, translated, reconstructed]
            classification = [['NEGATIVE', 'POSITIVE'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

            gen_imgs = np.concatenate(imgs)
            gen_imgs = 0.5 * gen_imgs + 0.5
            image_list.append(gen_imgs)
            if save_pics:
                correct_classification = ['NEGATIVE', 'POSITIVE', 'NEGATIVE'] if int(original_class) == 0 else [
                    'POSITIVE',
                    'NEGATIVE',
                    'POSITIVE']

                c = 3
                titles = ['Original', 'Translated', 'Reconstructed']
                fig, axs = plt.subplots(1, c, figsize=(15, 5))
                cnt = 0
                for j in range(c):
                    axs[j].imshow(gen_imgs[cnt], cmap='gray')
                    axs[j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                    axs[j].axis('off')
                    cnt += 1
                fig.savefig(f"predicted_{i}.png")
        return image_list
