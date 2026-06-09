"""
merge_data.py
-------------
Merges phyphox_df with HR CSV files.

HR filename convention: {subject}_{exercise}_{focus}_{set_nr}_hr.csv
  e.g. s_pushup_focus_1_hr.csv
       t_row_distracted_2_hr.csv
       s_squat_nf_3_hr.csv
"""

import os
import re
import numpy as np
import pandas as pd

encoder = {
    "subject":  {"S": 0, "T": 1},
    "exercise": {"push": 0, "row": 1, "sq": 2},
    "focus":    {"nf": 0, "f": 1},
}

EXERCISE_MAP = {
    "push": "push", "pushup": "push", "push-up": "push", "pushups": "push",
    "row": "row", "pullup": "row", "pull-up": "row", "australianpullup": "row",
    "sq": "sq", "squat": "sq", "squats": "sq",
}
FOCUS_MAP = {
    "f": "f", "focus": "f", "focused": "f",
    "nf": "nf", "nofocus": "nf", "distracted": "nf", "unfocused": "nf",
}


def parse_hr_filename(fname: str):
    """
    Parse HR CSV filename to (subject_int, exercise_int, focus_int, set_nr_int).
    Returns None if parsing fails.
    """
    stem = fname.lower().replace("_hr.csv", "").replace("-", "_")
    parts = stem.split("_")

    subj_key = parts[0].upper()
    if subj_key not in encoder["subject"]:
        return None

    rest = parts[1:]
    ex_int = foc_int = set_nr = None

    for ex_len in (3, 2, 1):
        if len(rest) < ex_len + 2:
            continue
        ex_candidate = "".join(rest[:ex_len])
        if ex_candidate not in EXERCISE_MAP:
            continue
        ex_int = encoder["exercise"][EXERCISE_MAP[ex_candidate]]

        for foc_len in (2, 1):
            if len(rest) < ex_len + foc_len + 1:
                continue
            foc_candidate = "".join(rest[ex_len:ex_len + foc_len])
            if foc_candidate not in FOCUS_MAP:
                continue
            foc_int = encoder["focus"][FOCUS_MAP[foc_candidate]]

            nr_token = rest[ex_len + foc_len]
            if re.fullmatch(r"\d+", nr_token):
                within_cond_nr = int(nr_token)
                set_nr = (within_cond_nr - 1) * 2 + (1 if foc_int == 1 else 2)
                break

        if foc_int is not None and set_nr is not None:
            break

    if None in (ex_int, foc_int, set_nr):
        return None

    return encoder["subject"][subj_key], ex_int, foc_int, set_nr


def interpolate_hr(hr_df: pd.DataFrame, n_samples: int = 1500, duration: float = 30.0) -> np.ndarray:
    """Interpolate HR onto a uniform 50 Hz grid of length n_samples, anchored at t=0."""
    hr_df = hr_df.copy().sort_values("time").reset_index(drop=True)
    t0 = hr_df["time"].iloc[0]
    hr_df["t_rel"] = (hr_df["time"] - t0).dt.total_seconds()

    target_t = np.linspace(0, duration, n_samples, endpoint=False)
    return np.interp(target_t, hr_df["t_rel"].values, hr_df["hr"].values)


def build_merged_df(phyphox_path: str, hr_dir: str) -> pd.DataFrame:
    ph = pd.read_csv(phyphox_path)
    ph = ph.drop(columns=["Unnamed: 0"], errors="ignore")
    ph["hr"] = np.nan

    hr_files = sorted(f for f in os.listdir(hr_dir) if f.endswith("_hr.csv"))
    matched, skipped = [], []

    for fname in hr_files:
        parsed = parse_hr_filename(fname)
        if parsed is None:
            print(f"  [SKIP – parse error] {fname}")
            skipped.append(fname)
            continue

        subj, ex, foc, s_nr = parsed

        hr_df = pd.read_csv(os.path.join(hr_dir, fname))
        hr_df["time"] = pd.to_datetime(hr_df["time"])

        mask = (
            (ph["subject"]  == subj) &
            (ph["exercise"] == ex)   &
            (ph["focus"]    == foc)  &
            (ph["set_nr"]   == s_nr)
        )
        n = mask.sum()
        if n == 0:
            print(f"  [SKIP – no match] {fname}  (decoded: S={subj} ex={ex} foc={foc} set={s_nr})")
            skipped.append(fname)
            continue

        ph.loc[mask, "hr"] = interpolate_hr(hr_df, n_samples=n)
        matched.append(fname)
        print(f"  [OK]  {fname}  →  subject={subj}, ex={ex}, focus={foc}, set={s_nr}  ({n} rows)")

    coverage = ph["hr"].notna().mean() * 100
    print(f"\nMatched {len(matched)}/{len(hr_files)} HR files.  Skipped: {skipped or 'none'}")
    print(f"HR coverage: {ph['hr'].notna().sum()}/{len(ph)} rows ({coverage:.1f}%)")
    return ph


if __name__ == "__main__":
    HR_DIR = "garmin"
    PHYPHOX = "phyphox_df"
    OUT = "merged_df.csv"

    merged = build_merged_df(PHYPHOX, HR_DIR)
    merged.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print(merged[merged["hr"].notna()].head())