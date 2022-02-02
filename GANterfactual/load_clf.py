from __future__ import print_function, division

# The trained classifier is loaded.
# Rewrite this function if you want to use another model architecture than our modified AlexNET.
# A model, which provides a 'predict' function, has to be returned.
from GANterfactual.mura_model import WristPredictNet
from configs.mura_pretraining_config import mura_config

M1_WEIGHTS_PATH = "../checkpoints/mura/best/cp.ckpt"
GPU_WEIGHTS_PATH = "checkpoints/mura_densenet/best/cp.ckpt"

config = mura_config

def load_classifier():
    model = WristPredictNet(config, train_base=config['train']['train_base'])
    model.built = True
    model.load_weights(M1_WEIGHTS_PATH)
    return model
