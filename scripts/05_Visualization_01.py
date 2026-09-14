import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


DATA_DIR = "/content/drive/MyDrive/MetS_Experiment/results"
OUT_DIR = "/content/drive/MyDrive/MetS_Experiment/results/figures"

# Google Drive mount
from google.colab import drive
drive.mount('/content/drive')


# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

MODEL_COLORS = {
    "DeepMLP": "#2E75B6",
    "Random_Forest": "#ED7D31",
    "Random Forest": "#ED7D31",
    "XGBoost": "#70AD47",
    "LightGBM": "#FFC000",
    "BasicMLP": "#7030A0",
    "Logistic_Regression": "#A5A5A5",
    "ResNetMLP": "#264478",
    "SVM": "#9E480E",
}


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# HE_obe leakage
def fig_heobe_leakage_auc(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "4_HEobe_leakage_before_after.csv"))
    models = df["Model"].unique().tolist()
    conditions = ["With HE_obe (Original/Leaky)", "Without HE_obe (Primary/Corrected)"]
    cond_style = {
        "With HE_obe (Original/Leaky)": {"color": "#C00000", "label": "With HE_obe (leaky)"},
        "Without HE_obe (Primary/Corrected)": {"color": "#2E75B6", "label": "Without HE_obe (corrected)"},
    }
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for i, cond in enumerate(conditions):
        sub = df[df["Condition"] == cond].set_index("Model").reindex(models)
        offset = (i - 0.5) * width
        style = cond_style[cond]
        bars = ax.bar(x + offset, sub["Weighted_AUC"], width,
                       label=style["label"], color=style["color"],
                       edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, sub["Weighted_AUC"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 0.95)
    ax.set_title("Model Performance With vs. Without HE_obe\n(Predictor–Outcome Overlap Check)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
    _save(fig, out_dir, "R1c4_R2c1_HEobe_leakage_auc.png")


def fig_heobe_shap_dominance(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "4_HEobe_shap_dominance.csv"))
    df = df.rename(columns={df.columns[0]: "Variable"}).sort_values("Mean_abs_SHAP")
    colors = ["#C00000" if v == "HE_obe" else "#8FAADC" for v in df["Variable"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(df["Variable"], df["Mean_abs_SHAP"], color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, df["Mean_abs_SHAP"]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("HE_obe Dominates SHAP Importance\n(Evidence of Predictor–Outcome Overlap)")
    legend_patch = mpatches.Patch(color="#C00000", label="HE_obe (diagnostic-overlap variable)")
    ax.legend(handles=[legend_patch], loc="lower right", frameon=False)
    _save(fig, out_dir, "R1c4_HEobe_shap_dominance.png")


# direct SHAP on the 7 ORIGINAL environmental variables (replaces the invalidated PCA/SPCA back-projection figure)
def fig_original_env_shap(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "8_shap_original_vars_top3.csv"))
    env_vars = ["PM10", "PM2_5", "NO2", "O3", "SO2", "CO", "green_1"]
    df = df[df["Variable"].isin(env_vars)]
    models = df["Model"].unique().tolist()

    pivot = df.pivot(index="Variable", columns="Model", values="Mean_abs_SHAP").reindex(env_vars)

    x = np.arange(len(env_vars))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, pivot[m], width, label=m,
               color=MODEL_COLORS.get(m, None), edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(env_vars)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Direct SHAP on the 7 Original Environmental Variables\n"
                 "(No PCA/SPCA Back-Projection)")
    ax.legend(frameon=False)
    _save(fig, out_dir, "R1c7_R2c2_original_env_shap.png")


# VIF check
def fig_vif_check(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "8_vif_check.csv")).sort_values("VIF", ascending=True)
    colors = ["#C00000" if v >= 10 else ("#ED7D31" if v >= 5 else "#548235") for v in df["VIF"]]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.barh(df["Variable"], df["VIF"], color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, df["VIF"]):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(5, color="orange", linestyle="--", linewidth=1, label="VIF = 5 (caution)")
    ax.axvline(10, color="red", linestyle="--", linewidth=1, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("Multicollinearity Among the 7 Original\nEnvironmental Variables (Re-checked)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "R1c16_vif_check.png")


# DeLong pairwise significance heatmap
def fig_delong_heatmap(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "1_delong_test_all_pairs.csv"))
    models = sorted(set(df["Comparison"].str.split(" vs ").str[0]) |
                     set(df["Comparison"].str.split(" vs ").str[1]))
    # Order by descending AUC (best model first) for readability
    auc_lookup = {}
    for _, row in df.iterrows():
        a, b = row["Comparison"].split(" vs ")
        auc_lookup[a] = row["AUC_1"]
        auc_lookup[b] = row["AUC_2"]
    models = sorted(models, key=lambda m: -auc_lookup.get(m, 0))

    n = len(models)
    pmat = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        a, b = row["Comparison"].split(" vs ")
        i, j = models.index(a), models.index(b)
        pmat[i, j] = row["p_value"]
        pmat[j, i] = row["p_value"]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    masked = np.ma.masked_invalid(pmat)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = pmat[i, j]
            txt = f"{p:.3f}" + ("*" if p < 0.05 else "")
            ax.text(j, i, txt, ha="center", va="center",
                     fontsize=8, fontweight=("bold" if p < 0.05 else "normal"),
                     color="black")
    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(models)
    ax.set_title("Pairwise DeLong Test p-values (8 Models, HE_obe Removed)\n* p < 0.05")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("p-value (capped at 0.20)")
    _save(fig, out_dir, "R1c8_R2c3_delong_pvalue_heatmap.png")


# class-imbalance correction: 4 conditions
def fig_class_imbalance_conditions(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "6_class_imbalance_4conditions_top3.csv"))
    metrics = ["AUC", "Precision", "Recall", "F1"]
    models = df["Model"].unique().tolist()
    conditions = df["Condition"].unique().tolist()

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=True)
    x = np.arange(len(conditions))
    width = 0.8 / len(metrics)
    colors = ["#2E75B6", "#ED7D31", "#70AD47", "#7030A0"]

    for ax, model in zip(axes, models):
        sub = df[df["Model"] == model].set_index("Condition").reindex(conditions)
        for i, metric in enumerate(metrics):
            offset = (i - (len(metrics) - 1) / 2) * width
            ax.bar(x + offset, sub[metric], width, label=metric, color=colors[i],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([c.split(". ")[-1] for c in conditions], rotation=20, ha="right", fontsize=8)
        ax.set_title(model)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Class-Imbalance Correction Strategy Comparison\n"
                 "(Raw / SMOTE only / Class-weight only / SMOTE + Class-weight)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out_dir, "R1c5_class_imbalance_4conditions.png")


# weighted vs. unweighted (survey weights)
def fig_weighted_vs_unweighted(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "5_weighted_vs_unweighted.csv"))
    models = df["Model"].unique().tolist()
    train_conditions = df["Training_Condition"].unique().tolist()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, 4) * width
    series = []
    for tc in train_conditions:
        sub = df[df["Training_Condition"] == tc].set_index("Model").reindex(models)
        series.append((f"{tc} / weighted eval", sub["AUC_weighted_eval"]))
        series.append((f"{tc} / unweighted eval", sub["AUC_unweighted_eval"]))

    colors = ["#2E75B6", "#9DC3E6", "#ED7D31", "#F8CBAD"]
    for off, (label, vals), c in zip(offsets, series, colors):
        ax.bar(x + off, vals, width, label=label, color=c, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.60, 0.70)
    ax.set_title("Effect of Survey-Weight Application\n(Training and Evaluation Combinations)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "R1c1_R2c9_weighted_vs_unweighted.png")


# Leave-One-Region-Out (LORO) geographic validation
def fig_loro_regional(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "7_LORO_top3_models.csv"))
    summary = pd.read_csv(os.path.join(data_dir, "7_LORO_summary_top3.csv"))
    models = df["Model"].unique().tolist()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    box_data, positions, labels = [], [], []
    pos = 0
    group_gap = 1.0
    width = 0.6
    for m in models:
        vals = df[df["Model"] == m]["AUC"].values
        box_data.append(vals)
        positions.append(pos)
        labels.append(m)
        pos += group_gap

    bp = ax.boxplot(box_data, positions=positions, widths=width, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(m, "#8FAADC"))
        patch.set_alpha(0.7)

    # overlay individual region points
    rng = np.random.default_rng(42)
    for p, vals in zip(positions, box_data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), p) + jitter, vals, s=14, color="black", alpha=0.5, zorder=3)

    # reference line: random-split AUC per model (from summary "mean" column is LORO mean;
    # random-split reference is documented in the letter — reuse 0_model_ranking_all8 if available)
    ranking_path = os.path.join(data_dir, "0_model_ranking_all8.csv")
    if os.path.exists(ranking_path):
        rk = pd.read_csv(ranking_path).set_index("Model")
        for p, m in zip(positions, models):
            if m in rk.index:
                ax.hlines(rk.loc[m, "AUC"], p - width / 2 - 0.05, p + width / 2 + 0.05,
                           color="red", linestyle="--", linewidth=1.3, zorder=4)
    ax.plot([], [], color="red", linestyle="--", label="Random-split AUC (reference)")

    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Leave-One-Region-Out (LORO) Cross-Validation\n(17 Regions, Dots = Individual Region AUC)")
    ax.legend(frameon=False)
    _save(fig, out_dir, "R1c6_LORO_regional_auc.png")


# subgroup performance (sex / age)
def fig_subgroup_auc(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "3_subgroup_top3.csv"))
    model_cols = {"AUC_DeepMLP": "DeepMLP", "AUC_Random_Forest": "Random_Forest", "AUC_XGBoost": "XGBoost"}
    subgroups = df["Subgroup"].tolist()

    x = np.arange(len(subgroups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (col, name) in enumerate(model_cols.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[col], width, label=name,
                       color=MODEL_COLORS.get(name), edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, df[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(subgroups, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 0.85)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Subgroup Performance by Sex and Age Group")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
    _save(fig, out_dir, "R1c17_subgroup_auc.png")


# green_1 contribution ratio vs. top-1 variable
def fig_green1_contribution(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "10_green1_contribution_ratio.csv"))
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax1.bar(x - width / 2, df["Top1_SHAP"], width, label="Top-1 variable SHAP",
                 color="#8FAADC", edgecolor="black", linewidth=0.4)
    b2 = ax1.bar(x + width / 2, df["green_1_SHAP"], width, label="green_1 SHAP",
                 color="#548235", edgecolor="black", linewidth=0.4)
    for b, v, top in zip(b1, df["Top1_SHAP"], df["Top1_Variable"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{top}\n{v:.3f}",
                  ha="center", va="bottom", fontsize=7.5)
    for b, v, pct in zip(b2, df["green_1_SHAP"], df["green_1_pct_of_top1"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}\n({pct:.1f}%)",
                  ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylabel("Mean |SHAP value|")
    ax1.set_title("green_1 Contribution Relative to the Top-1 Predictor\n(% = green_1 SHAP / Top-1 SHAP)")
    ax1.legend(frameon=False)
    _save(fig, out_dir, "R2c4_green1_contribution_ratio.png")


# threshold sweep (precision & recall curves)
def fig_threshold_sweep(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "11_threshold_sweep_top3.csv"))
    models = df["Model"].unique().tolist()

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in models:
        sub = df[df["Model"] == m].sort_values("Threshold")
        ax.plot(sub["Threshold"], sub["Precision"], marker="o", linestyle="-",
                color=MODEL_COLORS.get(m), label=f"{m} – Precision")
        ax.plot(sub["Threshold"], sub["Recall"], marker="s", linestyle="--",
                color=MODEL_COLORS.get(m), label=f"{m} – Recall", alpha=0.7)

    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision–Recall Trade-off Across Decision Thresholds")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, out_dir, "R1c15_threshold_sweep.png")


# all-8-model ranking after HE_obe removal
def fig_model_ranking_all8(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "0_model_ranking_all8.csv")).sort_values("AUC", ascending=True)
    colors = [MODEL_COLORS.get(m, "#8FAADC") for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(df["Model"], df["AUC"], color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, df["AUC"]):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.60, 0.70)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Revised Table 8: 8-Model Performance Ranking\n(HE_obe Removed, Corrected Primary Analysis)")
    _save(fig, out_dir, "R2c1_model_ranking_all8_revised_table8.png")


def run(data_dir, out_dir):
    """Generate every figure, reading CSVs from data_dir and saving PNGs to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        fig_heobe_leakage_auc,
        fig_heobe_shap_dominance,
        fig_original_env_shap,
        fig_vif_check,
        fig_delong_heatmap,
        fig_class_imbalance_conditions,
        fig_weighted_vs_unweighted,
        fig_loro_regional,
        fig_subgroup_auc,
        fig_green1_contribution,
        fig_threshold_sweep,
        fig_model_ranking_all8,
    ]

    print(f"Reading CSVs from : {data_dir}")
    print(f"Writing figures to: {out_dir}\n")
    if not os.path.isdir(data_dir):
        print(f"  !! DATA_DIR does not exist: {data_dir}\n"
              f"     Double-check the path (and that Google Drive is mounted, if used).")
        return

    for fn in tasks:
        print(f"[{fn.__name__}]")
        try:
            fn(data_dir, out_dir)
        except Exception as e:
            print(f"  !! failed: {e}")
    print("\nDone.")


def main():
    # Command-line overrides (optional) — lets you still run this from a
    # terminal as: python generate_response_letter_figures.py --data-dir ... --out-dir ...
    # In Colab/Jupyter just edit DATA_DIR / OUT_DIR above and call run() directly;
    # argparse is skipped automatically when no recognizable CLI args are passed.
    data_dir, out_dir = DATA_DIR, OUT_DIR
    argv = sys.argv[1:]
    if argv:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--data-dir", default=DATA_DIR,
                            help="Directory containing the saved *.csv result files")
        parser.add_argument("--out-dir", default=OUT_DIR,
                            help="Directory to write the generated PNG figures")
        args, _unknown = parser.parse_known_args(argv)
        data_dir, out_dir = args.data_dir, args.out_dir

    run(data_dir, out_dir)


if __name__ == "__main__":
    main()
