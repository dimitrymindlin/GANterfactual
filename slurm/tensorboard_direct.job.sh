#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_train_clf/alexNet/ \
    --name "alexNet for GANterfactual RSNA" \
    --description "This is the RSNA classifier from the GANterfactual paper." \
    --one_shot
