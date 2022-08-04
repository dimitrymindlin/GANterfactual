from __future__ import print_function, division

from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config
import sys
import tensorflow as tf

for arg in sys.argv:
    if arg == "--cycle_consistency":
        gan_config["train"]["cycle_consistency_loss_weight"] = 5
    if arg == "--cycle_consistency_original":
        gan_config["train"]["cycle_consistency_loss_weight"] = 10
    if arg == "--counterfactual":
        gan_config["train"]["counterfactual_loss_weight"] = 5
    if arg == "--discriminator":
        gan_config["train"]["discriminator_loss_weight"] = 5
    if arg == "--discriminator_max":
        gan_config["train"]["discriminator_loss_weight"] = 10
    if arg == "--times_generator":
        gan_config["train"]["generator_training_multiplier"] = 3
    if arg == "--resnet":
        gan_config["train"]["generator"] = "resnet"
    if arg == "--no_skip_connections":
        gan_config["train"]["skip_connections"] = False

if __name__ == '__main__':
    print("Test")
    print(len(tf.config.list_physical_devices('GPU')))
    print("Test")
    if len(tf.config.list_physical_devices('GPU')) == 0:
        TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
    else:
        TFDS_PATH = "../tensorflow_datasets"
    gan_config["train"]["tfds_path"] = TFDS_PATH
    gan = CycleGAN(gan_config)
    gan.construct()
    #gan.evaluate_clf()
    gan.train()
    gan.evaluate()
    # gan.save(os.path.join('..', 'models', 'GANterfactual'))
