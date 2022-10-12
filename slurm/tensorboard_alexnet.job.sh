#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/GANterfactual

tensorboard dev upload --logdir tensorboard_logs \
    --name "Classifiers for GANterfactual RSNA" \
    --description "This is the RSNA classifier from the GANterfactual paper." \
    --one_shot
