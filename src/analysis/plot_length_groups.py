"""Plot full-catalog HR@10 and NDCG@10 across user history-length quantile cohorts."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_length_groups(
    group_rows: list[dict],
    output_paths: list[str | Path] | None = None,
) -> list[str]:
    """Generate grouped bar charts comparing Popularity, GRU4Rec, and SASRec by history length."""
    if not group_rows:
        raise ValueError("group_rows must not be empty.")

    targets = [Path(p) for p in (output_paths or ["results/figures/length_quantiles.png"])]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)

    cohorts: list[tuple[int, str, str, int]] = []
    seen_cohorts: set[int] = set()
    for row in sorted(group_rows, key=lambda r: int(r["cohort_index"])):
        cidx = int(row["cohort_index"])
        if cidx not in seen_cohorts:
            seen_cohorts.add(cidx)
            cohorts.append((cidx, str(row["cohort_label"]), str(row["length_range"]), int(row["user_count"])))

    models = ["popularity", "gru4rec", "sasrec"]
    model_labels = {"popularity": "Most-Popular", "gru4rec": "GRU4Rec", "sasrec": "SASRec"}
    model_colors = {"popularity": "#dc2626", "gru4rec": "#0f766e", "sasrec": "#2563eb"}

    lookup = {(int(r["cohort_index"]), str(r["model"])): r for r in group_rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=160)
    x = np.arange(len(cohorts))
    width = 0.25

    x_tick_labels = [f"{label}\n({lrange}, n={cnt})" for _i, label, lrange, cnt in cohorts]

    for ax, metric_key, title in (
        (axes[0], "HR@10", "Test Full-Catalog HR@10 by History-Length Quantile"),
        (axes[1], "NDCG@10", "Test Full-Catalog NDCG@10 by History-Length Quantile"),
    ):
        max_val = 0.0
        for m_idx, model_name in enumerate(models):
            vals = [float(lookup[(cidx, model_name)][metric_key]) for cidx, *_ in cohorts]
            max_val = max(max_val, max(vals))
            offset = (m_idx - 1) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=model_colors[model_name],
                label=model_labels[model_name],
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.004,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_xticks(x, x_tick_labels, fontsize=8.8)
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        ax.set_ylabel(metric_key)
        ax.set_ylim(0, max_val * 1.22 if max_val > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    saved: list[str] = []
    for path in targets:
        fig.savefig(path)
        saved.append(str(path))
    plt.close(fig)
    return saved
