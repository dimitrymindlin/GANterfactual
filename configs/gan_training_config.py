gan_config = {
    "dataset": {
        "download": False,
    },
    "data": {
        "class_names": ["positive"],
        "input_size": (320, 320),
        "image_height": 320,
        "image_width": 320,
        "image_channel": 3,
    },
    "train": {
        "optimizer": "adam",
        "batch_size": 2,
        "learn_rate": 0.0001,
        "epochs": 30,
        "beta1": 0.5,
        "beta2": 0.999,
        "cycle_consistency_loss_weight": 8,
        "identity_loss_weight": 1,
        "counterfactual_loss_weight": 1,
        "wasserstein": False
    }
}
