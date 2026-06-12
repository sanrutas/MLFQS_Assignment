import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import title

from ML4QS.Python3Code.Chapter3.OutlierDetection import *
from ML4QS.Python3Code.Chapter3.DataTransformation import *

def detect_outliers(df):
    """
    Takes about 27 minutes to run on 36k dataset, so quite slow
    looking at 3d plots compared with Chavenet, seems to detect overall outliers slightly better
    That makes sense because takes into account all 3 dimensions at once.
    """

    lof_detector = DistanceBasedOutlierDetection()

    group_cols = ["subject", "set_nr", "exercise", "focus"]
    sensors = ["acc_lin", "acc", "gyro", "orientation"]

    sensor_cols = {
        sensor: (["yaw", "pitch", "roll"] if sensor == "orientation" else [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"])
        for sensor in sensors
    }
    df["_row_id"] = np.arange(len(df))

    df = df[(df["_row_id"] % 100 == 0)]
    #
    i = 0
    all_groups = []
    for _, df_group in df.groupby(group_cols, sort=False):
        for sensor, cols in sensor_cols.items():
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"progress: {i / 0.96:.2f} %") # 24 * 4 sensors
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            df_group = lof_detector.local_outlier_factor(
                df_group,
                cols,
                d_function="euclidean",
                k=10
            )
            # if i % 100 == 0:

            df_group = df_group.rename(columns={"lof": f"{sensor}_lof"})
            df_group[f"{sensor}_lof_outlier"] = df_group[f"{sensor}_lof"] > 2

            i += 1
        df_group = df_group[["_row_id"] + [f"{sensor}_lof_outlier" for sensor in sensors] + [f"{sensor}_lof" for sensor in sensors]]
        all_groups.append(df_group)

    full_lof_df = pd.concat(all_groups)

    df = df.merge(full_lof_df, on="_row_id", how="left")
    df = df.drop(columns="_row_id")

    for col in df.columns:
        if col.endswith("lof_outlier"):
            print(f"found {df[col].sum()} outliers for {col[:-len('lof_outlier') -1]} out of {len(df)} values ({df[col].sum() / len(df) * 100:.2f} %)")
    return df


def interpolate_outliers(df):
    sensors = ["acc_lin", "acc", "gyro", "orientation"]
    for sensor in sensors:
        if sensor == "orientation":
            cols = ["yaw", "pitch", "roll"]
        else:
            cols = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        df[cols] = df[cols].mask(df[f"{sensor}_lof_outlier"], np.nan)
        for col in cols:
            df[col] = df[col].interpolate()
    return df


def low_pass_smoothing(df, cutoff_frequency=5):
    """
    lpf should also be passed in parts because otherwise it would smooth boundaries. This goes quick though
    """
    lpf = LowPassFilter().low_pass_filter

    group_cols = ["subject", "set_nr", "exercise", "focus"]
    sensors = ["acc_lin", "acc", "gyro", "orientation"]

    sensor_cols = {
        sensor: (["yaw", "pitch", "roll"] if sensor == "orientation" else [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"])
        for sensor in sensors
    }
    df["_row_id"] = np.arange(len(df))
    i = 0
    all_groups = []
    for _, df_group in df.groupby(group_cols):
        for sensor, cols in sensor_cols.items():
            if sensor == "orientation":
                sensor_coord_cols = ["yaw", "pitch", "roll"]
            else:
                sensor_coord_cols = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]

            for col in sensor_coord_cols:
                df_group = lpf(data_table=df_group, col=col, sampling_frequency=50, cutoff_frequency=cutoff_frequency)

        lowpass_cols = [f"{col}_lowpass" for cols in sensor_cols.values() for col in cols]
        df_group = df_group[["_row_id"] + lowpass_cols]
        all_groups.append(df_group)

    full_lpf_df = pd.concat(all_groups)

    df = df.merge(full_lpf_df, on="_row_id", how="left")
    df = df.drop(columns="_row_id")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/merged_df.csv")
    df = detect_outliers(df)
    df = interpolate_outliers(df)
    df = low_pass_smoothing(df).drop("row_id")

    df.to_csv("data/df_processed.csv", index=False)
