from plots import *
import pandas as pd
from ML4QS.Python3Code.Chapter3.OutlierDetection import *

def summarize(df):

    numeric_cols = [
        "time",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "acc_lin_x", "acc_lin_y", "acc_lin_z",
        "yaw", "pitch", "roll",
        "hr"
    ]

    categorical_cols = [
        "subject",
        "exercise",
        "set_nr",
        "focus"
    ]

    num_summary = df[numeric_cols].describe().T.round(3)

    num_summary["missing"] = df[numeric_cols].isna().sum()
    num_summary["missing_pct"] = (
        df[numeric_cols].isna().mean() * 100
    ).round(2)

    num_summary["count"] = num_summary["count"].astype(int)

    num_summary = num_summary[
        [
            "count",
            "missing",
            "missing_pct",
            "min",
            "max",
            "mean",
            "std",
        ]
    ]

    cat_summary = df[categorical_cols].astype("category").describe().T

    cat_summary["missing"] = df[categorical_cols].isna().sum()
    cat_summary["missing_pct"] = (
        df[categorical_cols].isna().mean() * 100
    ).round(2)

    cat_summary = cat_summary[
        [
            "count",
            "missing",
            "missing_pct",
            "unique",
            "top",
            "freq",
        ]
    ]

    num_summary = num_summary.sort_values(
        by="missing_pct",
        ascending=False
    )

    cat_summary = cat_summary.sort_values(
        by="missing_pct",
        ascending=False
    )

    print("\nNUMERIC VARIABLES")
    print(num_summary)

    print("\nCATEGORICAL VARIABLES")
    print(cat_summary)

    num_latex = num_summary.to_latex(
        longtable=True,
        escape=True,
        caption="Numeric variable summary statistics",
        label="tab:numeric_summary_stats",
        float_format="%.3f"
    )

    with open("numeric_summary_stats.tex", "w") as f:
        f.write(num_latex)


    cat_latex = cat_summary.to_latex(
        longtable=True,
        escape=True,
        caption="Categorical variable summary statistics",
        label="tab:categorical_summary_stats"
    )

    with open("categorical_summary_stats.tex", "w") as f:
        f.write(cat_latex)

def explore_chauvenet_vs_lof(df):
    chauvenet = DistributionBasedOutlierDetection().chauvenet
    local_outlier_factor = DistanceBasedOutlierDetection().local_outlier_factor

    # thought to just take 1 example, but probably would be good
    # to make all nsets*3*4 time series plots and look for extreme anomalies with naked eye

    df = df[(df["set_nr"] == 1) & (df["exercise"] == 0) & (df["subject"] == 0)
        ][["time", "acc_lin_x", "acc_lin_y", "acc_lin_z"]]

    df_chau = df.copy()
    for axis in ["acc_lin_x", "acc_lin_y", "acc_lin_z"]:
        df_chau = chauvenet(data_table=df_chau, col=axis, C=2)
    df_chau["chau_outlier_xyz"] = df_chau["acc_lin_x_outlier"] | df_chau["acc_lin_y_outlier"] | df_chau[
        "acc_lin_z_outlier"]

    # looking into the local_outlier_factor function, z score normalizing for axes is done inside
    df_local_outlier_factor = local_outlier_factor(
        data_table=df, cols=['acc_lin_x', 'acc_lin_y', 'acc_lin_z'], k=10, d_function="euclidean"
    )
    df_local_outlier_factor["lof_outlier_xyz"] = df_local_outlier_factor["lof"] > 2
    plot_3d_outliers(
        df_chau,
        name="Chauvenet",
        mask_col="chau_outlier_xyz"
    )

    plot_3d_outliers(
        df_local_outlier_factor,
        name="LOF",
        mask_col="lof_outlier_xyz"
    )
    plot_xyz_outliers_over_time(df_local_outlier_factor, "LOF", "lof_outlier_xyz")
    plot_xyz_outliers_over_time(df_chau, "Chauvenet", "chau_outlier_xyz")

if __name__ == "__main__":
    df_merged = pd.read_csv("data/merged_df.csv")
    explore_chauvenet_vs_lof(df_merged)
    summarize(df_merged)
