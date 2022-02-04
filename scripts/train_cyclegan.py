from __future__ import print_function, division

import os
from GANterfactual.cyclegan import CycleGAN
from configs.mura_pretraining_config import mura_config

config = mura_config

if __name__ == '__main__':
    gan = CycleGAN()
    gan.construct(classifier_weight=1)
    gan.train(dataset_name=os.path.join("..", "data"), epochs=30, batch_size=mura_config["train"]["batch_size"],
              print_interval=10,
              sample_interval=100)
    gan.save(os.path.join('..', 'models', 'GANterfactual'))
