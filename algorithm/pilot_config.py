"""Validation for the preserved CPU/noisy-validation pilot configuration."""

import math


def validate_pilot_training_fields(config):
    """Keep the original pilot seed and method rules separate from final runs."""
    seeds = config.get("split_seeds", [])
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Training needs at least 1 distinct nonnegative split seeds")
    initial = config.get("initialization_seeds", [])
    if len(initial) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in initial):
        raise ValueError("Provide one nonnegative integer initialization seed per split seed")
    methods = config.get("methods", [])
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"forward", "gce", "ce"}:
        raise ValueError("Provide unique methods from forward, gce and ce")
    validate_pilot_config(config)


def validate_pilot_config(config):
    if config.get("train_fraction") != 0.8:
        raise ValueError("The assignment requires an exact 80/20 holdout")
    for field in ("epochs", "patience", "num_threads"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field, allow_zero in (("learning_rate", False), ("weight_decay", True), ("min_delta", True)):
        value = config.get(field)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"Invalid {field}")
    shuffle_seeds = config.get("shuffle_seeds", [])
    if (len(shuffle_seeds) != len(config["split_seeds"])
            or any(type(seed) is not int or seed < 0 for seed in shuffle_seeds)):
        raise ValueError("Provide one nonnegative integer shuffle seed per split seed")
    architecture = config.get("architecture", {})
    channels = architecture.get("channels", [])
    if (architecture.get("name") != "small_cnn" or architecture.get("num_classes") != 3
            or len(channels) != 2 or any(type(channel) is not int or channel <= 0 for channel in channels)):
        raise ValueError("Expected small_cnn with two positive channel widths and three classes")
    gamma = config.get("method_parameters", {}).get("gce", {}).get("gamma")
    if type(gamma) not in (int, float) or not 0 < gamma <= 1:
        raise ValueError("GCE gamma must lie in (0, 1]")
    for field in ("data_dir", "results_dir", "splits_dir"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError(f"Missing {field}")
    if config.get("optimizer") != "adam" or config.get("test_evaluation") != "disabled":
        raise ValueError("Pilots require Adam and disabled clean-test evaluation")
