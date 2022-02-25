gan_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["positive"],
        "input_size": (224, 224),
        "image_height": 224,
        "image_width": 224,
        "image_channel": 3,
    },
    "train": {
        "execution_id": None,
        "optimizer": "adam",
        "batch_size": 1,
        "learn_rate": 0.0001,
        "epochs": 30,
        "beta1": 0.5,
        "cycle_consistency_loss_weight": 1,
        "classifier_weight": 1,
        "counterfactual_loss_weight": 1,
        "clf_ckpt": "2022-02-25--10.22",
        "leaky_relu": False
    },
    "test": {
        "batch_size": 32,
    }
}
