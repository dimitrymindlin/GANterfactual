gan_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["negative", "positive"],
        "input_size": (512, 512),
        "image_height": 512,
        "image_width": 512,
        "image_channel": 3,
    },
    "train": {
        "execution_id": None,
        "optimizer": "adam",
        "batch_size": 1,
        "learn_rate": 0.0002,
        "epochs": 18,
        "beta1": 0.5,
        "adversarial_loss_weight": 1,
        "cycle_consistency_loss_weight": 10,
        "counterfactual_loss_weight": 1,
        "identity_loss_weight": 1,
        "clf_ckpt": "inception_mura/2022-06-04--00.05",
        "leaky_relu": True,
        "generator": "resnet",
        "skip_connections": True,
        "generator_training_multiplier": 1,
        "clf_model": "inception"
    },
    "test": {
        "batch_size": 10,
    }
}
