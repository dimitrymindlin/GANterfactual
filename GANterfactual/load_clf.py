from __future__ import print_function, division
from GANterfactual.mura_model import get_mura_model, get_finetuning_model_from_pretrained_model, get_working_mura_model
from configs.mura_pretraining_config import mura_config

M1_WEIGHTS_PATH = "../checkpoints/2022-02-25--10.22/cp.ckpt"


def load_classifier(gan_config):
    print("Loading Pretrained Model ...")
    GPU_WEIGHTS_PATH = f"checkpoints/{gan_config['train']['clf_ckpt']}/cp.ckpt"
    model = get_working_mura_model()
    #model = get_finetuning_model_from_pretrained_model(model)
    model.built = True
    model.load_weights(GPU_WEIGHTS_PATH).expect_partial()
    print("Model Loaded.")
    return model
