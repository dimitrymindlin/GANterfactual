from __future__ import print_function, division

from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config
import sys

for arg in sys.argv:
    if arg == "--cycle_consistency":
        gan_config["train"]["cycle_consistency_loss_weight"] = 10
    if arg == "--counterfactual":
        gan_config["train"]["counterfactual_loss_weight"] = 10
    if arg == "--four_times_generator":
        gan_config["train"]["generator_training_multiplier"] = 4
    if arg == "--resnet":
        gan_config["train"]["generator"] = "resnet"
    if arg == "--no_skip_connections":
        gan_config["train"]["skip_connections"] = False

if __name__ == '__main__':
    gan = CycleGAN(gan_config)
    gan.construct()
    # gan.evaluate_clf()
    gan.train()
    gan.evaluate()
    # gan.save(os.path.join('..', 'models', 'GANterfactual'))
