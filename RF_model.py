import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

from config import RF_SPACE

RECORD_COLS = ["set_id", "subject", "exercise", "set_nr", "focus"]

META_COLS = RECORD_COLS + [
    "time", "hr",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_lin_x", "acc_lin_y", "acc_lin_z",
    "yaw", "pitch", "roll",
    "acc_lin_lof", "acc_lin_lof_outlier",
    "acc_lof", "acc_lof_outlier",
    "gyro_lof", "gyro_lof_outlier",
    "orientation_lof", "orientation_lof_outlier",
    "acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass",
    "acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass",
    "gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass",
    "yaw_lowpass", "pitch_lowpass", "roll_lowpass",
    "acc_lin_magnitude", "acc_lin_sum", "acc_lin_product",
    "acc_magnitude", "acc_sum", "acc_product",
    "gyro_magnitude", "gyro_sum", "gyro_product",
    "orient_magnitude", "orient_sum", "orient_product",
]

DEFAULTS = {
    "n_estimators": 50,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1,
    "max_splits": 1296,
}


def get_splits(records, random_state=42, max_splits=9999):
    groups = []
    for subject in records["subject"].unique():
        for focus in records["focus"].unique():
            groups.append(records[
                (records["subject"] == subject) &
                (records["focus"] == focus)
            ].index.to_numpy())
    all_splits = [np.array(s) for s in product(*groups)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(all_splits)
    return all_splits[:min(max_splits, len(all_splits))]


def eval_over_splits(df, params, splits, cache=None, metric="balanced_accuracy"):
    """Called by tune.py. Returns (mean, std) of metric over splits."""
    cache = {} if cache is None else cache

    # build feature matrix once per unique feature set
    key = "rf_features"
    if key not in cache:
        data = pd.read_csv("data/featured_df.csv.gz")
        data["set_id"] = data.groupby(
            ["subject", "exercise", "set_nr", "focus"]).ngroup()
        feature_cols = [c for c in data.columns if c not in META_COLS]
        cache[key] = (data, feature_cols)

    data, feature_cols = cache[key]

    X = data[feature_cols]
    y = data["focus"]

    scores = []
    metric_fns = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    fn = metric_fns[metric]

    for test_idx in splits:
        test_mask  = data["set_id"].isin(test_idx)
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            min_samples_split=params["min_samples_split"],
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
        )
        rf.fit(X_train, y_train)

        # majority vote per set
        test_data = data[test_mask][["set_id", "focus"]].copy()
        test_data["pred"] = rf.predict(X_test)
        set_preds = test_data.groupby("set_id")["pred"].agg(
            lambda x: x.mode()[0])
        set_true = test_data.groupby("set_id")["focus"].first()

        scores.append(fn(set_true, set_preds))

    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    # quick sanity check
    df = pd.read_csv("data/featured_df.csv.gz")
    data = df.copy()
    data["set_id"] = data.groupby(
        ["subject", "exercise", "set_nr", "focus"]).ngroup()
    records = data[RECORD_COLS].drop_duplicates().reset_index(drop=True)
    splits = get_splits(records, max_splits=10)
    mean, std = eval_over_splits(df, DEFAULTS, splits)
    print(f"Sanity check (10 splits): {mean:.3f} +/- {std:.3f}")