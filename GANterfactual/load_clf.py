from __future__ import print_function, division
from GANterfactual.mura_model import get_mura_model, get_finetuning_model_from_pretrained_model
from configs.mura_pretraining_config import mura_config

M1_WEIGHTS_PATH = "../checkpoints/mura/best/cp.ckpt"


def load_classifier(gan_config):
    print("Loading Pretrained Model ...")
    GPU_WEIGHTS_PATH = f"checkpoints/{gan_config['train']['clf_ckpt']}/cp.ckpt"
    model = get_mura_model(mura_config)
    model = get_finetuning_model_from_pretrained_model(model)
    model.built = True
    model.load_weights(M1_WEIGHTS_PATH)
    print("Model Loaded.")
    return model
