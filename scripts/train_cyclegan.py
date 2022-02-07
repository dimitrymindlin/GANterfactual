from __future__ import print_function, division

import os
from GANterfactual.cyclegan import CycleGAN
from configs.gan_training_config import gan_config

config = gan_config

if __name__ == '__main__':
    gan = CycleGAN()
    gan.construct(classifier_weight=1)
    gan.train()
    gan.save(os.path.join('..', 'models', 'GANterfactual'))
