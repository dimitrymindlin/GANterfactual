#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate GANterfactual
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m GANterfactual/train_alexNet

