"""
make_plots.py
-------------
Generates RF result plots for the paper.
Run from project root after rf.py and rf_svm_features.py have completed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────
rf_wind = pd.read_csv("data/rf_record_results.csv")
rf_per  = pd.read_csv("data/rf_svm_features_record_results.csv")

svm_rec = pd.DataFrame({
    "set_id":       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    "subject":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "focus":        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "correct_rate": [0.991,0.449,1.000,0.995,0.069,0.991,0.000,1.000,1.000,1.000,
                     0.139,0.968,0.745,0.731,0.102,1.000,0.144,1.000,0.870,0.981,
                     0.167,0.162,0.037,0.912]
})

merged = rf_wind.merge(rf_per[["set_id","correct_rate"]], on="set_id",
                        suffixes=("_windowed","_per_rec"))
merged = merged.sort_values("correct_rate_per_rec", ascending=False).reset_index(drop=True)

ex_map    = {0: "Push-up", 1: "Pull-up", 2: "Squat"}
focus_map = {0: "Distracted", 1: "Focused"}
merged["Label"] = merged.apply(
    lambda r: f"S{int(r.subject)} {ex_map[int(r.exercise)][:4]} "
              f"{int(r.set_nr)} {focus_map[int(r.focus)][:4]}", axis=1)

# ── Plot 1: grouped bar chart — record-level correct rates ─────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
ax.set_facecolor("white")

x = np.arange(len(merged))
w = 0.35

ax.bar(x - w/2, merged["correct_rate_windowed"], w,
       label="RF Windowed", color="#4878CF", alpha=0.85)
ax.bar(x + w/2, merged["correct_rate_per_rec"],  w,
       label="RF Per-recording", color="#6ACC65", alpha=0.85)
ax.axhline(0.5, color="red", linestyle="--", linewidth=0.9, label="Chance (0.5)")

ax.set_xticks(x)
ax.set_xticklabels(merged["Label"], rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Correct Classification Rate")
ax.set_title("Record-level Classification Rate by Recording and Feature Representation")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("plots/rf_record_comparison.png", dpi=150)
plt.close()
print("Saved plots/rf_record_comparison.png")

# ── Plot 2: scatter — windowed vs per-rec, coloured by condition ───────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
ax.set_facecolor("white")

colors  = {"Focused": "#E8794B", "Distracted": "#4878CF"}
markers = {0: "o", 1: "s"}

for _, row in merged.iterrows():
    cond = focus_map[int(row["focus"])]
    ax.scatter(row["correct_rate_windowed"], row["correct_rate_per_rec"],
               color=colors[cond], marker=markers[int(row["subject"])],
               s=80, alpha=0.85, edgecolors="white", linewidth=0.5)

for cond, col in colors.items():
    ax.scatter([], [], color=col, label=cond, s=60)
ax.scatter([], [], marker="o", color="grey", label="Subject 0", s=60)
ax.scatter([], [], marker="s", color="grey", label="Subject 1", s=60)
ax.plot([0,1], [0,1], "k--", linewidth=0.8, alpha=0.4, label="Equal performance")
ax.axhline(0.5, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
ax.axvline(0.5, color="red", linestyle=":", linewidth=0.7, alpha=0.5)

ax.set_xlabel("Correct Rate — RF Windowed")
ax.set_ylabel("Correct Rate — RF Per-recording")
ax.set_title("Windowed vs Per-recording RF\nper recording")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plots/rf_scatter.png", dpi=150)
plt.close()
print("Saved plots/rf_scatter.png")

# ── Plot 3: mean correct rate by condition and subject across all 3 models ─
models = {
    "SVM\nPer-rec":    svm_rec,
    "RF\nWindowed":    rf_wind,
    "RF\nPer-rec":     rf_per,
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor="white")
for ax in axes:
    ax.set_facecolor("white")

x = np.arange(3)
w = 0.35
labels = list(models.keys())

# by condition
focused_means    = [df[df.focus==1]["correct_rate"].mean() for df in models.values()]
distracted_means = [df[df.focus==0]["correct_rate"].mean() for df in models.values()]

axes[0].bar(x - w/2, focused_means,    w, label="Focused",    color="#E8794B", alpha=0.85)
axes[0].bar(x + w/2, distracted_means, w, label="Distracted", color="#4878CF", alpha=0.85)
axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel("Mean Correct Rate")
axes[0].set_title("By Condition")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis="y", alpha=0.25)

# by subject
s0_means = [df[df.subject==0]["correct_rate"].mean() for df in models.values()]
s1_means = [df[df.subject==1]["correct_rate"].mean() for df in models.values()]

axes[1].bar(x - w/2, s0_means, w, label="Subject 0", color="#9467BD", alpha=0.85)
axes[1].bar(x + w/2, s1_means, w, label="Subject 1", color="#D62728", alpha=0.85)
axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_title("By Subject")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.25)

plt.suptitle("Mean Correct Classification Rate Across Models", fontsize=11)
plt.tight_layout()
plt.savefig("plots/rf_condition_subject_bars.png", dpi=150)
plt.close()
print("Saved plots/rf_condition_subject_bars.png")

print("\nAll plots saved to plots/")
