#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate GANterfactual
cd /media/compute/homes/dmindlin/GANterfactual

python3 -m scripts.train_cyclegan --discriminator --resnet --counterfactual