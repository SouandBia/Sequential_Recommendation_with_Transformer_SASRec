"""Plot validation HR@10 and NDCG@10 learning curves for SASRec vs baselines."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    curves_by_model: dict[str, list[dict]],
    popularity_val_metrics: dict[str, float] | None = None,
    output_paths: list[str | Path] | None = None,
) -> list[str]:
    """Generate side-by-side Validation NDCG@10 and HR@10 training curves."""
    if not curves_by_model:
        raise ValueError("curves_by_model must not be empty.")

    targets = [Path(p) for p in (output_paths or ["results/figures/validation_curves.png"])]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "sasrec": "#2563eb",
        "gru4rec": "#0f766e",
    }
    labels = {
        "sasrec": "SASRec (d=64, 2 blocks, 2 heads)",
        "gru4rec": "GRU4Rec (d=64, 1 layer)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)

    for ax, metric_key, title in (
        (axes[0], "NDCG@10", "Validation Full-Catalog NDCG@10"),
        (axes[1], "HR@10", "Validation Full-Catalog HR@10"),
    ):
        for model_name, history in curves_by_model.items():
            epochs = [int(row["epoch"]) for row in history]
            values = [float(row[metric_key]) for row in history]
            color = colors.get(model_name, "#7c3aed")
            label = labels.get(model_name, model_name)
            ax.plot(epochs, values, marker="o", markersize=3, linewidth=2.0, color=color, label=label)

            best_row = max(history, key=lambda r: float(r["NDCG@10"]))
            ax.scatter(
                [int(best_row["epoch"])],
                [float(best_row[metric_key])],
                color=color,
                edgecolors="black",
                s=45,
                zorder=5,
            )

        if popularity_val_metrics and metric_key in popularity_val_metrics:
            ax.axhline(
                float(popularity_val_metrics[metric_key]),
                color="#dc2626",
                linestyle="--",
                linewidth=1.6,
                label=f"Most-Popular ({popularity_val_metrics[metric_key]:.4f})",
            )

        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_key)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    saved: list[str] = []
    for path in targets:
        fig.savefig(path)
        saved.append(str(path))
    plt.close(fig)
    return saved
