from __future__ import print_function, division
from GANterfactual.mura_model import WristPredictNet
from configs.mura_pretraining_config import mura_config

config = mura_config

M1_WEIGHTS_PATH = "../checkpoints/mura/best/cp.ckpt"
GPU_WEIGHTS_PATH = f"checkpoints/{config['train']['clf_ckpt']}/cp.ckpt"


def load_classifier():
    print("Loading Pretrained Model ...")
    model = WristPredictNet(config, train_base=False)
    model.built = True
    model.load_weights(GPU_WEIGHTS_PATH)
    print("Model Loaded.")
    return model
