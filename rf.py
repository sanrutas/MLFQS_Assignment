import sys
sys.path.append('/Users/Thomas/ML4QS/Python3Code')  # adjust if needed

import warnings
import numpy as np
import pandas as pd
from itertools import product as itertools_product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from Chapter4.TemporalAbstraction import NumericalAbstraction

# ── Config ────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 1

AXIS_TRIPLES = {
    "acc_lin": ["acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass"],
    "acc":     ["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"],
    "gyro":    ["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"],
    "orient":  ["yaw_lowpass", "pitch_lowpass", "roll_lowpass"],
}

INDIVIDUAL_COLS = [
    "acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass",
    "acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass",
    "gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass",
    "yaw_lowpass", "pitch_lowpass", "roll_lowpass",
    "hr",
]

AGG_FUNCTIONS = ["mean", "std", "min", "max", "slope"]

META_COLS = [
    "time", "subject", "exercise", "set_nr", "focus", "hr",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "acc_lin_x", "acc_lin_y", "acc_lin_z", "yaw", "pitch", "roll",
    "acc_lin_lof", "acc_lin_lof_outlier", "acc_lof", "acc_lof_outlier",
    "gyro_lof", "gyro_lof_outlier", "orientation_lof", "orientation_lof_outlier",
    "acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass",
    "acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass",
    "gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass",
    "yaw_lowpass", "pitch_lowpass", "roll_lowpass",
    "acc_lin_magnitude", "acc_lin_sum", "acc_lin_product",
    "acc_magnitude", "acc_sum", "acc_product",
    "gyro_magnitude", "gyro_sum", "gyro_product",
    "orient_magnitude", "orient_sum", "orient_product",
    "set_id",
]

# ── Feature engineering ───────────────────────────────────────────────────
def add_axis_combinations(g):
    new_cols = []
    for name, (cx, cy, cz) in AXIS_TRIPLES.items():
        x, y, z = g[cx], g[cy], g[cz]
        g[f"{name}_magnitude"] = np.sqrt(x**2 + y**2 + z**2)
        g[f"{name}_sum"]       = x + y + z
        g[f"{name}_product"]   = x * y * z
        new_cols += [f"{name}_magnitude", f"{name}_sum", f"{name}_product"]
    return g, new_cols

def process_set(group):
    g = group.copy().reset_index(drop=True)
    g, combined_cols = add_axis_combinations(g)
    feature_cols = INDIVIDUAL_COLS + combined_cols
    g.index = pd.to_datetime(g["time"], unit="s")
    na = NumericalAbstraction()
    for agg in AGG_FUNCTIONS:
        g = na.abstract_numerical(g, feature_cols, WINDOW_SECONDS, agg)
    return g.reset_index(drop=True)

# ── Load featured df ────────────────────────────────────────────
print("Loading featured_df.csv.gz...")
full = pd.read_csv("data/featured_df.csv.gz")
print(f"Featured df shape: {full.shape}")

# ── RF exhaustive evaluation ──────────────────────────────────────────────
data = full.copy()
data["set_id"] = data.groupby(["subject", "exercise", "set_nr", "focus"]).ngroup()

set_meta = data.groupby("set_id")[["subject", "exercise", "set_nr", "focus"]].first().reset_index()

feature_cols = [c for c in data.columns if c not in META_COLS]
print(f"\n{len(feature_cols)} feature columns")

split_groups = []
for subject in sorted(data["subject"].unique()):
    for focus in sorted(data["focus"].unique()):
        candidates = set_meta[
            (set_meta["subject"] == subject) & (set_meta["focus"] == focus)
        ]["set_id"].values
        split_groups.append(candidates)

all_test_splits = list(itertools_product(*split_groups))
print(f"Total splits: {len(all_test_splits)}")

results_list = []
all_predictions = {sid: [] for sid in set_meta["set_id"]}

for i, test_set_ids in enumerate(all_test_splits):
    if i % 100 == 0:
        print(f"Split {i}/1296")

    test_mask  = data["set_id"].isin(test_set_ids)
    train_mask = ~test_mask

    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, "focus"]
    X_test  = data.loc[test_mask,  feature_cols]

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    test_data = data.loc[test_mask, ["set_id", "focus"]].copy()
    test_data["pred"] = rf.predict(X_test)

    set_preds = test_data.groupby("set_id")["pred"].agg(lambda x: x.mode()[0])
    set_true  = test_data.groupby("set_id")["focus"].first()

    acc  = accuracy_score(set_true, set_preds)
    bacc = balanced_accuracy_score(set_true, set_preds)
    prec = precision_score(set_true, set_preds, zero_division=0)
    rec  = recall_score(set_true, set_preds, zero_division=0)
    f1   = f1_score(set_true, set_preds, zero_division=0)

    results_list.append({'accuracy': acc, 'balanced_accuracy': bacc,
                         'precision': prec, 'recall': rec, 'f1': f1})

    for sid in test_set_ids:
        correct = int(set_preds[sid] == set_true[sid])
        all_predictions[sid].append(correct)

results_df = pd.DataFrame(results_list)
print("\n--- Results ---")
print("Mean:\n", results_df.mean().round(3))
print("Std:\n",  results_df.std().round(3))
results_df.to_csv("data/rf_results.csv", index=False)

record_results = []
for sid, preds in all_predictions.items():
    row = set_meta[set_meta["set_id"] == sid].iloc[0]
    record_results.append({
        "set_id": sid, "subject": row["subject"],
        "exercise": row["exercise"], "set_nr": row["set_nr"],
        "focus": row["focus"], "correct_rate": np.mean(preds)
    })

record_df = pd.DataFrame(record_results).sort_values("correct_rate", ascending=False)
record_df.to_csv("data/rf_record_results.csv", index=False)
print("\nRecord-level results:")
print(record_df.to_string(index=False))
print("\nDone. Results saved to data/rf_results.csv and data/rf_record_results.csv")