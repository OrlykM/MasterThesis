r"""
Reads every *_summary.json under comparisons/<dataset>/ (produced by
compare_reconstruction_3d.py) and generates one bar chart PER METRIC --
X axis is every (VSLAM+FDE, dataset) combination found, Y axis is that
metric's value. Saved as static PNG, ready to paste straight into a
presentation slide.

Usage:
    python3 plot_metrics_bars.py --comparisons_dir comparisons --output_dir metric_charts
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("ate_rmse_m", "ATE RMSE (м)", "01_ATE_RMSE"),
    ("rte_rmse_m", "RTE RMSE (м)", "02_RTE_RMSE"),
    ("rre_rmse_deg", "RRE RMSE (°)", "03_RRE_RMSE"),
    ("depth_abs_rel_sampled", "Depth AbsRel", "04_Depth_AbsRel"),
    ("recovered_scale", "Відновлений масштаб", "05_Vidnovlenyi_Masshtab"),
]


def load_all_summaries(root_dir):
    """Each immediate subfolder of root_dir = one dataset name; each
    *_summary.json inside it = one VSLAM+FDE combo's metrics for that dataset."""
    records = []
    for dataset_dir in sorted(d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)):
        dataset_name = os.path.basename(dataset_dir)
        for json_path in sorted(glob.glob(os.path.join(dataset_dir, "*_summary.json"))):
            with open(json_path) as f:
                data = json.load(f)
            vslam = data.get("vslam", "?")
            fde = data.get("fde", "?")
            label = f"{vslam}+{fde}\n@{dataset_name}"
            records.append({"label": label, "dataset": dataset_name, **data})
    return records


def plot_metric(records, metric_key, metric_title, output_path):
    pairs = [(r["label"], r[metric_key], f"{r.get('vslam','?')}+{r.get('fde','?')}")
             for r in records if metric_key in r and r[metric_key] is not None]
    if not pairs:
        print(f"  SKIP {metric_key}: no data found in any summary.json")
        return

    labels, values, pair_keys = zip(*pairs)

    # Sum this metric per (vslam+fde) pair ACROSS all datasets, find which
    # pair has the highest total and which has the lowest -- highlight every
    # bar belonging to those two pairs, wherever they appear on the x-axis.
    sums = {}
    for pk, v in zip(pair_keys, values):
        sums[pk] = sums.get(pk, 0.0) + v
    worst_pair = max(sums, key=sums.get)
    best_pair = min(sums, key=sums.get)

    colors = []
    for pk in pair_keys:
        if pk == worst_pair:
            colors.append("#D62728")  # red -- highest summed value across datasets
        elif pk == best_pair:
            colors.append("#2CA02C")  # green -- lowest summed value across datasets
        else:
            colors.append("#4C72B0")  # default blue

    fig_width = max(10, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(range(len(labels)), values, color=colors)

    ax.bar_label(bars, fmt=lambda v: f"{v:.2f}", padding=3, fontsize=8)
    ymax = max(values)
    ax.set_ylim(top=ymax * 1.12 if ymax > 0 else 1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_title)
    ax.set_title(f"{metric_title}: VSLAM x FDE x Dataset")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#D62728", label=f"найбільша сумарна похибка: {worst_pair}"),
        Patch(facecolor="#2CA02C", label=f"найменша сумарна похибка: {best_pair}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}  ({len(labels)} стовпчиків) -- "
          f"найгірша пара: {worst_pair} (сума={sums[worst_pair]:.2f}), "
          f"найкраща пара: {best_pair} (сума={sums[best_pair]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons_dir", required=True,
                     help="e.g. comparisons/ (contains run1/, run2/, ... subfolders, each with *_summary.json)")
    ap.add_argument("--output_dir", default="metric_charts")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_all_summaries(args.comparisons_dir)
    print(f"Знайдено {len(records)} записів (комбінація VSLAM+FDE+датасет) у {args.comparisons_dir}")
    if not records:
        print("Нічого не знайдено -- перевір шлях і що там реально лежать *_summary.json файли.")
        return

    for key, title, filename in METRICS:
        plot_metric(records, key, title, os.path.join(args.output_dir, f"{filename}.png"))


if __name__ == "__main__":
    main()