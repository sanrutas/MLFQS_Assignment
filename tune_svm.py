"""Standalone Bayesian tuning for the SVM (C, gamma) with Optuna.

Kept separate from tune.py because the SVM refits PCA inside every split
(so there is nothing to cache) and needs prepare_svm_df run once up front.
"""

import numpy as np
import pandas as pd
import optuna

import SVM_model as SVM
import config


def _all_splits(records, seed):
    groups = []
    for subject in records["subject"].unique():
        for focus in records["focus"].unique():
            groups.append(records[(records["subject"] == subject) &
                                  (records["focus"] == focus)].index.to_numpy())
    import itertools
    splits = [np.array(s) for s in itertools.product(*groups)]
    np.random.default_rng(seed).shuffle(splits)
    return splits


def partition_splits(data, n_tune=40, n_eval=40, seed=0):
    records = data[SVM.META_COLS].drop_duplicates().reset_index(drop=True)
    s = _all_splits(records, seed)
    return s[:n_tune], s[n_tune:n_tune + n_eval]


def suggest_from_space(trial, space):
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return out


def run(data, n_trials=40, n_tune=40, n_eval=40, split_seed=0, sampler_seed=0,
        metric="balanced_accuracy"):
    """data must already be the per-set feature table (prepare_svm_df output)."""
    data = data.reset_index(drop=True)
    tune_splits, eval_splits = partition_splits(data, n_tune, n_eval, split_seed)

    def objective(trial):
        params = {**SVM.DEFAULTS, **suggest_from_space(trial, config.SVM_SPACE)}
        mean, std = SVM.eval_over_splits(data, params, tune_splits, None, metric)
        trial.set_user_attr("std", std)
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    print(f"[svm] {n_trials} trials x {len(tune_splits)} tune splits (TPE, metric={metric})")

    def _cb(st, tr):
        print(f"  trial {tr.number:>3}: {metric}={tr.value:.3f} (best={st.best_value:.3f})  "
              + "  ".join(f"{k}={v:.4g}" for k, v in tr.params.items()))

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    return study, tune_splits, eval_splits


def results_table(study, metric="balanced_accuracy"):
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = dict(t.params)
        row[f"{metric}_mean"] = round(t.value, 4)
        row["std"] = round(t.user_attrs.get("std", float("nan")), 4)
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values(f"{metric}_mean", ascending=False)
            .reset_index(drop=True))


def best_params(study):
    return {**SVM.DEFAULTS, **study.best_params}


if __name__ == "__main__":
    # build the per-set feature table ONCE
    data = SVM.prepare_svm_df(pd.read_csv("data/df_processed.csv"))

    study, tune_splits, eval_splits = run(data, n_trials=30, n_tune=40, n_eval=40)

    print("\n=== top configs (TUNE splits) ===")
    print(results_table(study).head(10).to_string(index=False))

    best = best_params(study)
    mean, std = SVM.eval_over_splits(data, best, eval_splits, None, "balanced_accuracy")
    print(f"\n[svm] HELD-OUT balanced_accuracy: {mean:.3f} +/- {std:.3f}")
    print("[svm] best params:", study.best_params)