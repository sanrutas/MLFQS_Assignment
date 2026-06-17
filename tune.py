
import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import optuna

import LSTM_model as LSTM
import SVM_model as SVM
import RF_model as RF
import config


def _all_splits(records, seed):
    groups = []
    for subject in records["subject"].unique():
        for focus in records["focus"].unique():
            groups.append(records[(records["subject"] == subject) &
                                  (records["focus"] == focus)].index.to_numpy())
    splits = [np.array(s) for s in itertools.product(*groups)]
    np.random.default_rng(seed).shuffle(splits)
    return splits


def partition_splits(df, n_tune=40, n_eval=40, seed=0):
    records = df[LSTM.RECORD_COLS].drop_duplicates().reset_index(drop=True)
    s = _all_splits(records, seed)
    return s[:n_tune], s[n_tune:n_tune + n_eval]


def suggest_from_space(trial, space):
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                          step=spec.get("step", 1))
        elif t == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                            log=spec.get("log", False))
        else:
            raise ValueError(f"unknown space type {t!r} for {name!r}")
    return out


# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    space: dict
    base_params: dict
    eval_fn: Callable
    direction: str = "maximize"


LSTM_tune = ModelSpec("lstm", config.LSTM_SPACE, dict(LSTM.PARAMS), LSTM.eval_over_splits)
SVM_tune = ModelSpec("svm", config.SVM_SPACE, dict(SVM.DEFAULTS), SVM.eval_over_splits)
RF_tune = ModelSpec("rf", config.RF_SPACE, dict(RF.DEFAULTS), RF.eval_over_splits)

def run(df, spec, n_trials=40, n_tune=40, n_eval=40, split_seed=0,
        sampler_seed=0, metric="balanced_accuracy", verbose=True):
    tune_splits, eval_splits = partition_splits(df, n_tune, n_eval, split_seed)
    cache = {}                                   # shared -> windows/features reused

    def objective(trial):
        params = {**spec.base_params, **suggest_from_space(trial, spec.space)}
        mean, std = spec.eval_fn(df, params, tune_splits, cache, metric)
        trial.set_user_attr("std", std)
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=spec.direction, sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    if verbose:
        print(f"[{spec.name}] {n_trials} trials x {len(tune_splits)} tune splits "
              f"(TPE, metric={metric})")

        def _cb(st, tr):
            print(f"  trial {tr.number:>3}: {metric}={tr.value:.3f} "
                  f"(best={st.best_value:.3f})  "
                  + "  ".join(f"{k}={v}" for k, v in tr.params.items()))
        callbacks = [_cb]
    else:
        callbacks = []

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    return study, tune_splits, eval_splits


def results_table(study, metric="balanced_accuracy"):
    """Trials sorted best-first, with the per-config std."""
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = dict(t.params)
        row[f"{metric}_mean"] = round(t.value, 4)
        row["std"] = round(t.user_attrs.get("std", float("nan")), 4)
        rows.append(row)
    asc = study.direction == optuna.study.StudyDirection.MINIMIZE
    return (pd.DataFrame(rows)
            .sort_values(f"{metric}_mean", ascending=asc)
            .reset_index(drop=True))


def best_params(study, spec):
    return {**spec.base_params, **study.best_params}


if __name__ == "__main__":
    df = LSTM.add_axis_combinations(pd.read_csv("data/df_processed.csv"))

    for spec in (RF_tune,):
        study, tune_splits, eval_splits = run(
            df, spec, n_trials=30, n_tune=30, n_eval=30, split_seed=0,
        )
        print(f"\n=== {spec.name}: top configs (on TUNE splits) ===")
        print(results_table(study).head(10).to_string(index=False))

        best = best_params(study, spec)
        mean, std = spec.eval_fn(df, best, eval_splits, {}, "balanced_accuracy")
        print(f"\n[{spec.name}] HELD-OUT balanced_accuracy: {mean:.3f} +/- {std:.3f}")
        print(f"[{spec.name}] best params:", study.best_params)