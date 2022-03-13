from __future__ import print_function, division
from GANterfactual.mura_model import WristPredictNet
from configs.mura_pretraining_config import mura_config
import tensorflow as tf
import tensorflow_addons as tfa

M1_WEIGHTS_PATH = "../checkpoints/2022-02-25--10.22/cp.ckpt"


def load_classifier(gan_config):
    print("Loading Pretrained Model ...")
    GPU_WEIGHTS_PATH = f"checkpoints/{gan_config['train']['clf_ckpt']}/cp.ckpt"
    model = WristPredictNet(mura_config)
    # model = get_finetuning_model_from_pretrained_model(model)
    model.built = True
    model.load_weights(GPU_WEIGHTS_PATH).expect_partial()
    print("Model Loaded.")
    return model


def load_classifier_complete(gan_config):
    print("Loading Pretrained Model ...")
    metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    model = tf.keras.models.load_model(f"checkpoints/{gan_config['train']['clf_ckpt']}/model", custom_objects={'f1_score': metric_f1})
    print("Model Loaded.")
    return model
