import numpy as np
import pandas as pd
from itertools import product
from scipy.fft import rfft, rfftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

META_COLS = ["set_id", "subject", "exercise", "set_nr", "focus"]
TARGET_COL = "focus"
PARAMS = {
    "gamma":"scale",
    "C":1
}


def add_axis_combinations(g):
    AXIS_TRIPLES = {
        "acc_lin": ["acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass"],
        "acc":     ["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"],
        "gyro":    ["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"],
        "orient":  ["yaw_lowpass", "pitch_lowpass", "roll_lowpass"],
    }
    new_data = {}

    for name, (cx, cy, cz) in AXIS_TRIPLES.items():
        x, y, z = g[cx], g[cy], g[cz]

        new_data[f"{name}_magnitude"] = np.sqrt(x**2 + y**2 + z**2)
        # new_data[f"{name}_sum"] = x + y + z
        new_data[f"{name}_abs_sum"] = x.abs() + y.abs() + z.abs()
        eps = 1e-8
        new_data[f"{name}_pairwise_product_sum"] = x*y + x*z + y*z


    new_df = pd.DataFrame(new_data, index=g.index)
    return pd.concat([g, new_df], axis=1), list(new_data.keys())

def get_fft_df(df):

    SAMPLE_RATE = 50
    sensor_cols = [c for c in df.columns if c not in ["time", "subject", "exercise", "set_nr", "focus", "set_id"]]

    def fft_features(x):
        x = np.asarray(x)

        fft_vals = np.abs(rfft(x))
        freqs = rfftfreq(len(x), d=1 / SAMPLE_RATE)

        fft_vals[0] = 0

        dominant_idx = np.argmax(fft_vals)

        dominant_freq = freqs[dominant_idx]
        dominant_power = fft_vals[dominant_idx]

        p = fft_vals / (fft_vals.sum() + 1e-12)

        spectral_entropy = -(p * np.log2(p + 1e-12)).sum()
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12)

        return {
            "dominant_freq": dominant_freq,
            "dominant_power": dominant_power,
            "spectral_entropy": spectral_entropy,
            "spectral_centroid": spectral_centroid,
        }


    fft_rows = []

    for set_id, group in df.groupby("set_id"):
        row = {
            "set_id": set_id,
            "subject": group["subject"].iloc[0],
            "exercise": group["exercise"].iloc[0],
            "set_nr": group["set_nr"].iloc[0],
            "focus": group["focus"].iloc[0],
        }

        for col in sensor_cols:
            feats = fft_features(group[col])

            for feat_name, value in feats.items():
                row[f"{col}_{feat_name}"] = value

        fft_rows.append(row)

    return pd.DataFrame(fft_rows)

def prepare_svm_df(df):
    fft_df = get_fft_df(df)

    df, newcols = add_axis_combinations(df)
    sensor_cols = [c for c in df.columns if c not in ["time", "subject", "exercise", "set_nr", "focus", "set_id"]]
    agg_df = df.groupby("set_id")[sensor_cols].agg(["std", "min", "max", "median"])
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    svm_df = agg_df.merge(fft_df, on="set_id", how="inner")

    return svm_df

def get_splits(data, random_state=42, max_splits=9999):
    split_groups = []
    for subject in data["subject"].unique():
        for focus in data["focus"].unique():
            candidates = data[(data["subject"] == subject) &
                              (data["focus"] == focus)].index.to_numpy()
            split_groups.append(candidates)
    all_test_splits = [np.array(split) for split in product(*split_groups)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(all_test_splits)
    return all_test_splits[:min(max_splits, len(all_test_splits))]


def get_feature_groups(feature_cols):
    return {
        "acc_lin": [c for c in feature_cols if c.startswith("acc_lin_")],
        "acc":     [c for c in feature_cols if c.startswith("acc_") and not c.startswith("acc_lin_")],
        "gyro":    [c for c in feature_cols if c.startswith("gyro_")],
        "orient":  [c for c in feature_cols if c.startswith(("yaw_", "pitch_", "roll_", "orient_"))],
        "hr":      [c for c in feature_cols if c.startswith("hr_")],
    }


def pca_per_group(X_train_raw, X_test_raw, groups, n_train):
    """Scale + PCA each sensor group separately (fit on train, apply to test).
    Returns (train_pca_df, test_pca_df) — same columns as your inline version."""
    train_pca_df = pd.DataFrame(index=X_train_raw.index)
    test_pca_df = pd.DataFrame(index=X_test_raw.index)

    for group_name, cols in groups.items():
        if len(cols) == 0:
            continue
        n_comp = 3 if group_name == "hr" else 6
        n_comp = min(n_comp, len(cols), n_train - 1)

        scaler = StandardScaler()
        pca = PCA(n_components=n_comp)
        X_train_group_scaled = scaler.fit_transform(X_train_raw[cols])
        X_test_group_scaled = scaler.transform(X_test_raw[cols])
        train_pcs = pca.fit_transform(X_train_group_scaled)
        test_pcs = pca.transform(X_test_group_scaled)

        for i in range(n_comp):
            train_pca_df[f"{group_name}_pc{i+1}"] = train_pcs[:, i]
            test_pca_df[f"{group_name}_pc{i+1}"] = test_pcs[:, i]

    return train_pca_df, test_pca_df


def build_extra_features(train_df_raw, test_df_raw):
    train_extra = pd.get_dummies(train_df_raw[["exercise", "set_nr"]],
                                 columns=["exercise"], prefix="exercise", dtype=int)
    test_extra = pd.get_dummies(test_df_raw[["exercise", "set_nr"]],
                                columns=["exercise"], prefix="exercise", dtype=int)
    test_extra = test_extra.reindex(columns=train_extra.columns, fill_value=0)
    return train_extra, test_extra


def build_design_matrices(train_df_raw, test_df_raw, feature_cols):
    X_train_raw = train_df_raw[feature_cols]
    X_test_raw = test_df_raw[feature_cols]

    groups = get_feature_groups(feature_cols)
    train_pca_df, test_pca_df = pca_per_group(X_train_raw, X_test_raw, groups,
                                              n_train=len(train_df_raw))
    train_extra, test_extra = build_extra_features(train_df_raw, test_df_raw)

    X_train = pd.concat([train_pca_df, train_extra], axis=1)
    X_test = pd.concat([test_pca_df, test_extra], axis=1)
    return X_train, X_test


def score_predictions(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def run_single_split(data, test_idx, repeat, feature_cols, C=1.0, gamma="scale"):
    train_idx = data.index.difference(test_idx).to_numpy()
    train_df_raw = data.loc[train_idx]
    test_df_raw = data.loc[test_idx]

    X_train, X_test = build_design_matrices(train_df_raw, test_df_raw, feature_cols)
    y_train = train_df_raw[TARGET_COL]
    y_test = test_df_raw[TARGET_COL]

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=PARAMS["C"], gamma=PARAMS["gamma"], class_weight="balanced")),
    ])
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    result = {
        "repeat": repeat,
        **score_predictions(y_test, y_pred),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }

    pred_df = test_df_raw[META_COLS].copy()
    pred_df["repeat"] = repeat
    pred_df["pred"] = y_pred
    pred_df["correct"] = pred_df["focus"] == pred_df["pred"]
    return result, pred_df


def run_repeated_cv(data, selected_test_splits, n_repeats=100, C=1.0, gamma="scale"):
    """Loop the splits, collecting metrics and per-set predictions."""
    feature_cols = [c for c in data.columns if c not in META_COLS]
    results, all_predictions = [], []

    for repeat in range(n_repeats):
        if repeat % 50 == 0:
            print(f"progress: {repeat / n_repeats * 100:.0f}%")
        test_idx = selected_test_splits[repeat]
        result, pred_df = run_single_split(data, test_idx, repeat, feature_cols,
                                           C=C, gamma=gamma)
        results.append(result)
        all_predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    return results_df, predictions_df


def summarize(results_df, predictions_df):
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    print(results_df[metric_cols].describe())
    print("\nMean accuracy:", results_df["accuracy"].mean())
    print("Mean balanced accuracy:", results_df["balanced_accuracy"].mean())
    print("\nPrediction counts by record:")
    print(
        predictions_df
        .groupby(META_COLS)["correct"]
        .agg(["count", "mean"])
        .sort_values("mean")
    )


DEFAULTS = {"C": 1.0, "gamma": "scale", "seed": 42}


def eval_over_splits(df, p, splits, cache=None, metric="balanced_accuracy"):
    df = df.reset_index(drop=True)               # align with harness split indices
    feature_cols = [c for c in df.columns if c not in META_COLS]
    C = p.get("C", 1.0)
    gamma = p.get("gamma", "scale")

    scores = []
    for repeat, test_idx in enumerate(splits):
        result, _ = run_single_split(df, test_idx, repeat, feature_cols, C=C, gamma=gamma)
        scores.append(result[metric])
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    data = prepare_svm_df(pd.read_csv('data/df_processed.csv'))
    selected_test_splits = get_splits(data)
    results_df, predictions_df = run_repeated_cv(data, selected_test_splits, n_repeats=1296)
    summarize(results_df, predictions_df)