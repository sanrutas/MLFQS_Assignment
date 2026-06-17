import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)


RECORD_COLS = ["set_id", "subject", "exercise", "set_nr", "focus"]

from config import LSTM_TUNABLE_PARAMS

STABLE = {
    "bidirectional": True,
    "pooling": "meanmax",
    "head_hidden": 64,
    "use_exercise": True,
    "batch_size": 32,
    "max_splits": 50,
    "seed": 42,
}

PARAMS = {**STABLE, **LSTM_TUNABLE_PARAMS}


# --------------------------------------------------------------------------- #
# 1.  Build OVERLAPPING WINDOWS instead of one giant 1500-step sequence.
#     Each window keeps its parent set's label; predictions are aggregated
#     back to the set at inference time.
# --------------------------------------------------------------------------- #
def make_windows(df, p):
    sensor_cols = [c for c in df.columns if c not in RECORD_COLS + ["time"]]
    records = df[RECORD_COLS].drop_duplicates().reset_index(drop=True)
    ex_codes = pd.factorize(records["exercise"])[0]  # robust 0..k-1 codes

    Xw, win_rec, win_exercise = [], [], []
    for ridx, row in records.iterrows():
        g = df[df["set_id"] == row["set_id"]].sort_values("time")
        arr = g[sensor_cols].to_numpy(dtype=np.float32)
        if p["decim"] > 1:  # safe: data is already low-pass filtered
            arr = arr[::p["decim"]]
        L = arr.shape[0]
        for s in range(0, max(1, L - p["window"] + 1), p["stride"]):
            w = arr[s:s + p["window"]]
            if w.shape[0] < p["window"]:
                continue
            Xw.append(w)
            win_rec.append(ridx)
            win_exercise.append(int(ex_codes[ridx]))

    return (records, sensor_cols,
            np.stack(Xw),
            np.asarray(win_rec, dtype=np.int64),
            np.asarray(win_exercise, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 2.  Your splitting procedure, unchanged: test = 1 focused + 1 unfocused
#     per subject (4 sets), the rest train.
# --------------------------------------------------------------------------- #
def get_splits(records, random_state=42, max_splits=9999):
    groups = []
    for subject in records["subject"].unique():
        for focus in records["focus"].unique():
            groups.append(records[(records["subject"] == subject) &
                                  (records["focus"] == focus)].index.to_numpy())
    all_splits = [np.array(s) for s in product(*groups)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(all_splits)
    return all_splits[:min(max_splits, len(all_splits))]


# --------------------------------------------------------------------------- #
# 3.  Plain LSTM. Pooling over all timesteps (default) keeps the one change
#     that actually fixed the original model, without any conv/GRU machinery.
# --------------------------------------------------------------------------- #
class LSTMClassifier(nn.Module):
    def __init__(self, n_sensors, n_exercise, p):
        super().__init__()
        H = p["hidden_size"]
        self.bidirectional = p["bidirectional"]
        self.pooling = p["pooling"]
        self.use_exercise = p["use_exercise"]
        D = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=H,
            num_layers=p["num_layers"],
            batch_first=True,
            dropout=p["dropout"] if p["num_layers"] > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        feat_dim = D * H * (2 if self.pooling == "meanmax" else 1)
        in_dim = feat_dim + (n_exercise if self.use_exercise else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, p["head_hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
            nn.Linear(p["head_hidden"], 1),
        )

    def forward(self, x_seq, x_ex):
        out, (h_n, _) = self.lstm(x_seq)  # out: (B, T, D*H)
        if self.pooling == "meanmax":
            z = torch.cat([out.mean(1), out.max(1).values], dim=1)
        else:  # "last": final hidden state
            z = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        if self.use_exercise:
            z = torch.cat([z, x_ex], dim=1)
        return self.head(z).squeeze(1)


def _augment(x, p):
    """Jitter + per-channel scaling. Train only."""
    x = x + torch.randn_like(x) * p["aug_noise"]
    s = (torch.rand(x.size(0), 1, x.size(2), device=x.device) - 0.5) * 2 * p["aug_scale"]
    return x * (1 + s)


def train_model(Xw, ex_oh, yw, n_sensors, p, device):
    ds = TensorDataset(torch.tensor(Xw), torch.tensor(ex_oh), torch.tensor(yw))
    loader = DataLoader(ds, batch_size=p["batch_size"], shuffle=True)
    model = LSTMClassifier(n_sensors, ex_oh.shape[1], p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    pos = yw.sum();
    neg = len(yw) - pos  # class weighting for window imbalance
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(p["n_epochs"]):
        for xb, eb, yb in loader:
            xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
            xb = _augment(xb, p)
            opt.zero_grad()
            loss = loss_fn(model(xb, eb), yb)
            loss.backward()
            opt.step()
    return model


def run_single_split(Xw, win_rec, win_ex, records, y_rec, test_idx,
                     n_sensors, n_exercise, p, device):
    train_recs = records.index.difference(test_idx).to_numpy()
    tr = np.isin(win_rec, train_recs)
    te = np.isin(win_rec, test_idx)

    seq_len, n_feat = Xw.shape[1], Xw.shape[2]
    scaler = StandardScaler().fit(Xw[tr].reshape(-1, n_feat))  # fit on TRAIN windows only
    Xtr = scaler.transform(Xw[tr].reshape(-1, n_feat)).reshape(-1, seq_len, n_feat).astype(np.float32)
    Xte = scaler.transform(Xw[te].reshape(-1, n_feat)).reshape(-1, seq_len, n_feat).astype(np.float32)

    eye = np.eye(n_exercise, dtype=np.float32)
    ex_tr, ex_te = eye[win_ex[tr]], eye[win_ex[te]]
    y_tr = y_rec[win_rec[tr]].astype(np.float32)

    model = train_model(Xtr, ex_tr, y_tr, n_sensors, p, device)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte).to(device), torch.tensor(ex_te).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    # aggregate windows -> per-set prediction (mean probability)
    set_prob = pd.Series(probs).groupby(win_rec[te]).mean()
    set_pred = (set_prob >= 0.5).astype(int)
    y_true = y_rec[set_prob.index.to_numpy()]

    return {
        "accuracy": accuracy_score(y_true, set_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, set_pred),
        "precision": precision_score(y_true, set_pred, zero_division=0),
        "recall": recall_score(y_true, set_pred, zero_division=0),
        "f1": f1_score(y_true, set_pred, zero_division=0),
    }, list(zip(set_prob.index.to_numpy(), y_true, set_pred.to_numpy(), set_prob.to_numpy()))


def main(df, params=None):
    p = dict(PARAMS if params is None else params)  # copy so tuning callers don't mutate
    torch.manual_seed(p["seed"]);
    np.random.seed(p["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records, sensor_cols, Xw, win_rec, win_ex = make_windows(df, p)
    y_rec = records["focus"].to_numpy(dtype=np.float32)
    n_exercise = int(win_ex.max()) + 1
    n_sensors = Xw.shape[2]

    print(f"device={device}  sensors={n_sensors}  windows={Xw.shape}  "
          f"({Xw.shape[0] / len(records):.0f} windows/set)")

    splits = get_splits(records, p["seed"], p["max_splits"])
    results, preds = [], []
    for r, ti in enumerate(splits):
        if r % 10 == 0:
            print(f"  split {r}/{len(splits)}")
        res, pr = run_single_split(Xw, win_rec, win_ex, records, y_rec, ti,
                                   n_sensors, n_exercise, p, device)
        results.append(res);
        preds.extend(pr)

    res_df = pd.DataFrame(results)
    pred_df = pd.DataFrame(preds, columns=["record", "focus", "pred", "prob"])
    pred_df["correct"] = pred_df["focus"] == pred_df["pred"]
    print("\n=== mean over %d splits ===" % len(res_df))
    print(res_df.mean().round(3).to_dict())
    return res_df, pred_df


def eval_over_splits(df, p, splits, cache=None, metric="balanced_accuracy"):
    """Average `metric` over the given splits for one LSTM config.
    Returns (mean, std). `cache` (a dict) reuses windowed arrays across configs
    that share window/stride/decim. Used by the tuning harness."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(p["seed"]);
    np.random.seed(p["seed"])

    cache = {} if cache is None else cache
    key = ("lstm_win", p["window"], p["stride"], p["decim"])
    if key not in cache:
        cache[key] = make_windows(df, p)
    records, _, Xw, win_rec, win_ex = cache[key]
    y_rec = records["focus"].to_numpy(dtype=np.float32)
    n_exercise = int(win_ex.max()) + 1
    n_sensors = Xw.shape[2]

    scores = [run_single_split(Xw, win_rec, win_ex, records, y_rec, ti,
                               n_sensors, n_exercise, p, device)[0][metric]
              for ti in splits]
    return float(np.mean(scores)), float(np.std(scores))


def add_axis_combinations(df):
    TRIPLES = {
        "acc_lin": ["acc_lin_x_lowpass", "acc_lin_y_lowpass", "acc_lin_z_lowpass"],
        "acc": ["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"],
        "gyro": ["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"],
        "orient": ["yaw_lowpass", "pitch_lowpass", "roll_lowpass"],
    }
    nd = {}
    for name, (cx, cy, cz) in TRIPLES.items():
        x, y, z = df[cx], df[cy], df[cz]
        nd[f"{name}_magnitude"] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        nd[f"{name}_abs_sum"] = x.abs() + y.abs() + z.abs()
        nd[f"{name}_pairwise_product_sum"] = x * y + x * z + y * z
    return pd.concat([df, pd.DataFrame(nd, index=df.index)], axis=1)


if __name__ == "__main__":
    df = pd.read_csv("data/df_processed.csv")
    df = add_axis_combinations(df)

    results_df, predictions_df = main(df)  # uses PARAMS

    print("\nPer-set hit-rate:")
    print(predictions_df.groupby("record")["correct"].mean().sort_values())