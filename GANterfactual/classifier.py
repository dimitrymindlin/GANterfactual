from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import Activation, ZeroPadding2D, Lambda
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf

# The trained classifier is loaded.
# Rewrite this function if you want to use another model architecture than our modified AlexNET.
# A model, which provides a 'predict' function, has to be returned.
from GANterfactual.mura_model import WristPredictNet
from configs.mura_pretraining_config import mura_config

config = mura_config

def load_classifier():
    model = WristPredictNet(config, train_base=config['train']['train_base'])
    model.built = True
    model.load_weights("../checkpoints/mura/best/cp.ckpt")
    return model
