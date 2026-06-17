from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_3d_outliers(df, name, mask_col, cols=("acc_lin_x", "acc_lin_y", "acc_lin_z")):
    x_col, y_col, z_col = cols

    mask = df[mask_col].astype(bool)
    t = np.arange(len(df))

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    normal = ax.scatter(
        df.loc[~mask, x_col],
        df.loc[~mask, y_col],
        df.loc[~mask, z_col],
        c=t[~mask],
        cmap="viridis",
        s=10,
        alpha=0.45,
        label="Normal"
    )

    ax.scatter(
        df.loc[mask, x_col],
        df.loc[mask, y_col],
        df.loc[mask, z_col],
        marker="x",
        color="red",
        s=80,
        linewidths=2,
        label="Outliers",
        zorder=10
    )

    cbar = fig.colorbar(normal, ax=ax, pad=0.1)
    cbar.set_label("Time / sample index")

    ax.set_title(f"{name}: 3D XYZ Outlier Detection")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{mask_col}_plot_3d")


def plot_xyz_outliers_over_time(df, name, mask_col):
    cols = ["acc_lin_x", "acc_lin_y", "acc_lin_z"]
    mask = df[mask_col].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, facecolor="white")

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], alpha=0.8)
        ax.scatter(
            df.index[mask],
            df.loc[mask, col],
            marker="x",
            color="red",
            s=70,
            linewidths=2,
            label="Outliers"
        )
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)

    axes[0].set_title(f"{name}: XYZ Outliers Over Time")
    axes[-1].set_xlabel("Time / sample")
    axes[0].legend()

    plt.tight_layout()
    plt.savefig(f"plots/{mask_col}_plot_ts")


def plot_importance(importance_df, top=20, path=None):

    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = importance_df.head(top).iloc[::-1]   # most important at top
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"],
            color="#4C72B0", ecolor="#999", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Permutation importance")
    # ax.set_title(f"SVM PCA importance (top {min(top, len(importance_df))})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        print("saved", path)
    return fig



if __name__ == "__main__":
    # svm_importance = pd.read_csv('results/svm/importance.csv')
    # plot_importance(svm_importance, path='results/svm/SVM_importances')

    svm_importance = pd.read_csv('results/lstm/tuned_run1/importance.csv')
    plot_importance(svm_importance, path='results/lstm/tuned_run1/LSTM_importances')