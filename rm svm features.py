"""
rf_svm_features.py
------------------
Runs RF on exactly Saturnas's feature set (per-recording FFT + descriptive
stats + PCA) with the same 1296 exhaustive splits, to isolate model vs
feature contribution vs the windowed RF.
"""

import numpy as np
import pandas as pd
from itertools import product as itertools_product
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading df_processed.csv...")
df = pd.read_csv("data/df_processed.csv")
df = df[["time", "subject", "exercise", "set_nr", "focus", "hr",
         *[col for col in df.columns if col.endswith("lowpass")]]]
df["set_id"] = df.groupby(["subject", "exercise", "set_nr", "focus"]).ngroup()

# ── FFT features (one row per recording) ──────────────────────────────────
SAMPLE_RATE = 50

sensor_cols = [c for c in df.columns
               if c not in ["time", "subject", "exercise", "set_nr", "focus", "set_id"]]

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
    return {"dominant_freq": dominant_freq, "dominant_power": dominant_power,
            "spectral_entropy": spectral_entropy, "spectral_centroid": spectral_centroid}

print("Computing FFT features...")
fft_rows = []
for set_id, group in df.groupby("set_id"):
    row = {"set_id": set_id, "subject": group["subject"].iloc[0],
           "exercise": group["exercise"].iloc[0], "set_nr": group["set_nr"].iloc[0],
           "focus": group["focus"].iloc[0]}
    for col in sensor_cols:
        for feat_name, value in fft_features(group[col]).items():
            row[f"{col}_{feat_name}"] = value
    fft_rows.append(row)
fft_df = pd.DataFrame(fft_rows)

# ── Axis combinations ─────────────────────────────────────────────────────
AXIS_TRIPLES = {
    "acc_lin": ["acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass"],
    "acc":     ["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"],
    "gyro":    ["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"],
    "orient":  ["yaw_lowpass", "pitch_lowpass", "roll_lowpass"],
}

for name, (cx, cy, cz) in AXIS_TRIPLES.items():
    x, y, z = df[cx], df[cy], df[cz]
    df[f"{name}_magnitude"] = np.sqrt(x**2 + y**2 + z**2)
    df[f"{name}_sum"]       = x + y + z
    df[f"{name}_abs_sum"]   = x.abs() + y.abs() + z.abs()

# ── Aggregate to one row per recording ────────────────────────────────────
sensor_cols_full = [c for c in df.columns
                    if c not in ["time", "subject", "exercise", "set_nr", "focus", "set_id"]]

agg_df = df.groupby("set_id")[sensor_cols_full].agg(["std", "min", "max", "median"])
agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
agg_df = agg_df.reset_index()

svm_df = agg_df.merge(fft_df, on="set_id", how="inner")
print(f"Feature matrix: {svm_df.shape}")

# ── Split structure ───────────────────────────────────
meta_cols = ["set_id", "subject", "exercise", "set_nr", "focus"]

split_groups = []
for subject in sorted(svm_df["subject"].unique()):
    for focus in sorted(svm_df["focus"].unique()):
        candidates = svm_df[
            (svm_df["subject"] == subject) & (svm_df["focus"] == focus)
        ].index.to_numpy()
        split_groups.append(candidates)

all_test_splits = list(itertools_product(*split_groups))
print(f"Total splits: {len(all_test_splits)}")

# ── PCA groups ────────────────────────────────────────
feature_cols = [c for c in svm_df.columns if c not in meta_cols]

pca_groups = {
    "acc_lin": [c for c in feature_cols if c.startswith("acc_lin_")],
    "acc":     [c for c in feature_cols if c.startswith("acc_") and not c.startswith("acc_lin_")],
    "gyro":    [c for c in feature_cols if c.startswith("gyro_")],
    "orient":  [c for c in feature_cols if c.startswith(("yaw_", "pitch_", "roll_", "orient_"))],
    "hr":      [c for c in feature_cols if c.startswith("hr_")],
}

# ── Exhaustive evaluation ─────────────────────────────────────────────────
results_list = []
all_predictions = {sid: [] for sid in svm_df["set_id"]}

for i, test_idx in enumerate(all_test_splits):
    test_idx = np.array(test_idx)  # flatten tuple to array
    if i % 100 == 0:
        print(f"Split {i}/1296")

    train_idx = svm_df.index.difference(test_idx).to_numpy()

    train_raw = svm_df.loc[train_idx]
    test_raw  = svm_df.loc[test_idx]

    y_train = train_raw["focus"]
    y_test  = test_raw["focus"]

    # PCA per sensor group, fitted on train only
    train_pca = pd.DataFrame(index=train_raw.index)
    test_pca  = pd.DataFrame(index=test_raw.index)

    for group_name, cols in pca_groups.items():
        if not cols:
            continue
        n_comp = 3 if group_name == "hr" else 6
        n_comp = min(n_comp, len(cols), len(train_raw) - 1)

        scaler = StandardScaler()
        pca    = PCA(n_components=n_comp)

        Xtr = scaler.fit_transform(train_raw[cols])
        Xte = scaler.transform(test_raw[cols])

        tr_pcs = pca.fit_transform(Xtr)
        te_pcs = pca.transform(Xte)

        for j in range(n_comp):
            train_pca[f"{group_name}_pc{j+1}"] = tr_pcs[:, j]
            test_pca[f"{group_name}_pc{j+1}"]  = te_pcs[:, j]

    # exercise one-hot + set_nr controls
    train_extra = pd.get_dummies(train_raw[["exercise", "set_nr"]],
                                 columns=["exercise"], prefix="exercise", dtype=int)
    test_extra  = pd.get_dummies(test_raw[["exercise", "set_nr"]],
                                 columns=["exercise"], prefix="exercise", dtype=int)
    test_extra  = test_extra.reindex(columns=train_extra.columns, fill_value=0)

    X_train = pd.concat([train_pca, train_extra], axis=1)
    X_test  = pd.concat([test_pca,  test_extra],  axis=1)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results_list.append({"accuracy": acc, "balanced_accuracy": bacc,
                          "precision": prec, "recall": rec, "f1": f1})

    for sid, pred, true in zip(test_raw["set_id"], y_pred, y_test):
        all_predictions[sid].append(int(pred == true))

results_df = pd.DataFrame(results_list)
print("\n--- Results ---")
print("Mean:\n", results_df.mean().round(3))
print("Std:\n",  results_df.std().round(3))
results_df.to_csv("data/rf_svm_features_results.csv", index=False)

record_results = []
for sid, preds in all_predictions.items():
    row = svm_df[svm_df["set_id"] == sid].iloc[0]
    record_results.append({
        "set_id": sid, "subject": row["subject"],
        "exercise": row["exercise"], "set_nr": row["set_nr"],
        "focus": row["focus"], "correct_rate": np.mean(preds)
    })

record_df = pd.DataFrame(record_results).sort_values("correct_rate", ascending=False)
record_df.to_csv("data/rf_svm_features_record_results.csv", index=False)
print("\nRecord-level results:")
print(record_df.to_string(index=False))
print("\nDone.")