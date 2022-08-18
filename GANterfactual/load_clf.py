from __future__ import print_function, division
import tensorflow as tf

#M1_WEIGHTS_PATH = "../checkpoints/2022-03-24--12.42/model"


"""def load_classifier(gan_config):
    print("Loading Pretrained Model ...")
    GPU_WEIGHTS_PATH = f"checkpoints/{gan_config['train']['clf_ckpt']}/cp.ckpt"
    model = WristPredictNet(mura_config)
    # model = get_finetuning_model_from_pretrained_model(model)
    model.built = True
    model.load_weights(GPU_WEIGHTS_PATH).expect_partial()
    print("Model Loaded.")
    return model"""


def load_classifier_complete(gan_config):
    print(f"Loading Pretrained Model ... checkpoints/{gan_config['train']['clf_ckpt']}/model")
    model = tf.keras.models.load_model(f"checkpoints/{gan_config['train']['clf_ckpt']}/model", compile=False)
    print("Model Loaded.")
    return model
