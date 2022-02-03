#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate GANterfactual

cd /media/compute/homes/dmindlin/GANterfactual

tensorboard dev upload --logdir logs \
    --name "GAN Training" \
    --description "Cyclegan trained on mura pics" \
    --one_shot
