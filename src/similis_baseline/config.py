from copy import deepcopy


DEFAULT_TEMPLATE_CONFIG = {
    "thresholds": {
        "object_type": 0.0,
        "integrity": 0.0,
        "part_zone": 0.60,
        "material_group": 0.65,
    }
}


DEFAULT_CONFIG = {
    "seed": 42,
    "fields": ["object_type", "integrity", "material_group", "part_zone"],
    "group_key_column": "group_key",
    "backbone": "resnet18",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 0,
    "epochs": 4,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "label_smoothing": 0.0,
    "use_class_weights": True,
    "train_aug": True,
    "tiny_subset_size": 48,
    "tiny_epochs": 12,
    "device": "cpu",
    "template_config": DEFAULT_TEMPLATE_CONFIG,
}


def get_default_config():
    return deepcopy(DEFAULT_CONFIG)
