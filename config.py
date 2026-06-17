
LSTM_TUNABLE_PARAMS = {
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.484,
    "n_epochs": 100,
    "lr": 0.00293,
    "weight_decay": 0.00722,
    "window": 200,
    "stride": 75,
    "decim": 2,
    "aug_noise": 0.05,
    "aug_scale": 0.116,
}

LSTM_SPACE = {
    # "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
    # "n_epochs":    {"type": "int",   "low": 40, "high": 150, "step": 10},
    "dropout":     {"type": "float", "low": 0.1, "high": 0.5},
    "lr":          {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
    "weight_decay":{"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "window":      {"type": "categorical", "choices": [150, 200, 300]},
    "aug_scale":   {"type": "float", "low": 0.0, "high": 0.15},
}

SVM_SPACE = {
    "C":     {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
    "gamma": {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
}