from datetime import datetime

from tensorflow.keras import Input
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from GANterfactual.dataloader import DataLoader
from GANterfactual.discriminator import build_discriminator
from GANterfactual.generator import build_generator
import tensorflow as tf
import os
import numpy as np

from GANterfactual.load_clf import load_classifier
from configs.mura_pretraining_config import mura_config

execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
writer = tf.summary.create_file_writer(f'logs/' + execution_id)


class CycleGAN():
    def __init__(self, gan_config):
        self.img_rows = gan_config["data"]["image_height"]
        self.img_cols = gan_config["data"]["image_width"]
        self.channels = gan_config["data"]["image_channel"]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.wasserstein = gan_config["train"]["wasserstein"]
        self.gan_config = gan_config
        self.gan_config['train']['execution_id'] = execution_id
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = self.gan_config["train"]["cycle_consistency_loss_weight"]  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        self.d_N = None
        self.d_P = None
        self.g_NP = None
        self.g_PN = None
        self.combined = None
        self.classifier = None

    def construct(self, load_clf=True, classifier_weight=None):
        # Build the discriminators
        self.d_N = build_discriminator(self.img_shape, self.df)
        self.d_P = build_discriminator(self.img_shape, self.df)

        # Build the generators
        self.g_NP = build_generator(self.img_shape, self.gf, self.channels, relu=self.gan_config['train']['leaky_relu'])
        self.g_PN = build_generator(self.img_shape, self.gf, self.channels)

        self.build_combined(load_clf, classifier_weight)

    def load_existing(self, cyclegan_folder, load_clf=True, classifier_weight=None):
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

        self.build_combined(load_clf, classifier_weight)

    def save(self, cyclegan_folder):
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d_N.save(os.path.join(cyclegan_folder, 'discriminator_n.h5'))
        self.d_P.save(os.path.join(cyclegan_folder, 'discriminator_p.h5'))

        # Save generators to disk
        self.g_NP.save(os.path.join(cyclegan_folder, 'generator_np.h5'))
        self.g_PN.save(os.path.join(cyclegan_folder, 'generator_pn.h5'))

    def build_combined(self, load_clf=True, classifier_weight=None):
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
        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)
        # Identity mapping of images
        # img_N_id = self.g_PN(img_N)
        # img_P_id = self.g_NP(img_P)

        # For the combined model we will only train the generators
        self.d_N.trainable = False
        self.d_P.trainable = False

        # Discriminators determines validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        if load_clf:
            self.classifier = load_classifier(self.gan_config)
            self.classifier._name = "classifier"
            self.classifier.trainable = False

            # Counterfactual loss?
            class_N_loss = self.classifier(fake_N)
            class_P_loss = self.classifier(fake_P)

            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           class_N_loss, class_P_loss,
                                           reconstr_N, reconstr_P])
            self.combined.compile(loss=['mse', 'mse',
                                        'mse', 'mse',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                classifier_weight, classifier_weight,
                                                self.lambda_cycle, self.lambda_cycle],
                                  optimizer=optimizer)

        else:
            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           reconstr_N, reconstr_P])

            self.combined.compile(loss=['mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                self.lambda_cycle, self.lambda_cycle],
                                  optimizer=optimizer)

    def train(self, print_interval=100, sample_interval=1000):
        config_matrix = [[k, str(w)] for k, w in self.gan_config["train"].items()]
        with writer.as_default():
            tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

        batch_size = self.gan_config["train"]["batch_size"]
        epochs = self.gan_config["train"]["epochs"]
        data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), config=mura_config)

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        class_N = np.stack([np.ones(batch_size), np.zeros(batch_size)]).T
        class_P = np.stack([np.zeros(batch_size), np.ones(batch_size)]).T

        for epoch in range(epochs):
            for batch_i, (imgs_N, imgs_P) in enumerate(data_loader.load_batch()):
                # ----------------------
                #  Train Discriminators
                # ----------------------i

                # Translate images to opposite domain
                # P = class label 1, N = class label 0
                fake_P = self.g_NP.predict(imgs_N)
                fake_N = self.g_PN.predict(imgs_P)

                # Train the discriminators (original images = real / translated = Fake)
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

                if self.classifier is not None:
                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           class_N, class_P,
                                                           imgs_N, imgs_P])
                else:
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           imgs_N, imgs_P])

                if self.classifier is not None:
                    if batch_i % print_interval == 0:
                        with writer.as_default():
                            tf.summary.scalar('D_loss', tf.reduce_sum(d_loss[0]), step=epoch)
                            # the accuracy of a counterfactual image generator is the percentage of
                            # counterfactuals that actually changed the classifier’s prediction
                            tf.summary.scalar('acc', tf.reduce_sum(100 * d_loss[1]), step=epoch)
                            tf.summary.scalar('G_loss', tf.reduce_sum(g_loss[0]), step=epoch)
                            # cycle consistency loss
                            tf.summary.scalar('adv', tf.reduce_sum(np.mean(g_loss[1:3])), step=epoch)
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

    def predict(self, force_original_aspect_ratio=False):

        assert (self.classifier is not None)
        data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), config=mura_config)

        original, original_class = data_loader.load_single()
        x = tf.expand_dims(original, 0)
        # original = original.reshape(1, original.shape[0], original.shape[1], original.shape[2])

        pred_original = self.classifier.predict(x)
        classification = int(np.argmax(pred_original))
        if classification == 0:
            classification_label = "NEGATIVE"
            print("PREDICTION -- NEGATIVE")
            translated = self.g_NP.predict(x)
            reconstructed = self.g_PN.predict(translated)
        else:
            classification_label = "POSITIVE"
            print("PREDICTION -- POSITIVE")
            translated = self.g_PN.predict(x)
            reconstructed = self.g_NP.predict(translated)

        pred_translated = self.classifier.predict(translated)
        pred_reconstructed = self.classifier.predict(reconstructed)

        translated = translated[0]
        reconstructed = reconstructed[0]

        # data_loader.save_single(translated, translated_out_path)
        # data_loader.save_single(reconstructed, reconstructed_out_path)
        imgs = [original, translated, reconstructed]

        # gen_imgs = np.concatenate(imgs)
        gen_imgs = imgs
        correct_classification = ['NEGATIVE', 'POSITIVE', 'NEGATIVE'] if int(original_class) == 0 else ['POSITIVE',
                                                                                                        'NEGATIVE',
                                                                                                        'POSITIVE']

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        c = 3
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(1, c, figsize=(15, 5))
        cnt = 0
        for j in range(c):
            axs[j].imshow(gen_imgs[cnt], cmap='gray')
            axs[j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification_label})')
            axs[j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
            axs[j].axis('off')
            cnt += 1
        fig.savefig("predicted.png")
        plt.close()

        return [pred_original, pred_translated, pred_reconstructed]
