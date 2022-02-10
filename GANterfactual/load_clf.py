from __future__ import print_function, division
from GANterfactual.mura_model import WristPredictNet
from configs.mura_pretraining_config import mura_config

M1_WEIGHTS_PATH = "../checkpoints/mura/best/cp.ckpt"


def load_classifier(gan_config):
    print("Loading Pretrained Model ...")
    GPU_WEIGHTS_PATH = f"checkpoints/{gan_config['train']['clf_ckpt']}/cp.ckpt"
    model = WristPredictNet(mura_config, train_base=False)
    model.built = True
    model.load_weights(GPU_WEIGHTS_PATH)
    print("Model Loaded.")
    return model
