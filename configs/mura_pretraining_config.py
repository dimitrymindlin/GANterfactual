mura_config = {
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
        "train_base": False,
        "augmentation": False,
        "use_class_weights": True,
        "batch_size": 1,
        "epochs": 30,
        "learn_rate": 0.0001,
        "patience_learning_rate": 1,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8
    },
    "test": {
        "batch_size": 8,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "inception",
        "pretrained": True,
        "pooling": "max",
    }
}
