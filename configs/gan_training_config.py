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
        "epochs": 15,
        "beta1": 0.5,
        "cycle_consistency_loss_weight": 1,
        "identity_loss_weight": 1,
        "counterfactual_loss_weight": 10,
        "clf_ckpt": "2022-03-14--22.35",
        "leaky_relu": True,
        "generator": "unet",
        "skip_connections": True,
        "generator_training_multiplier": 1
    },
    "test": {
        "batch_size": 10,
    }
}
