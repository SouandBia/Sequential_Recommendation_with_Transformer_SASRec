"""Plot MovieLens-1M rating and user sequence-length distributions for Report Section 1."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dataset_distributions(
    stats: dict,
    output_dir: str | Path = "report/figures",
) -> dict[str, str]:
    """Generate rating histogram and user sequence-length distribution figures."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rating_counts: dict[int, int] = {int(k): int(v) for k, v in stats["rating_counts"].items()}
    lengths = np.asarray(stats["all_user_lengths"], dtype=np.int64)

    # 1. Rating distribution bar chart
    ratings_fig_path = out_path / "ml1m_rating_distribution.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    stars = sorted(rating_counts.keys())
    counts = [rating_counts[s] for s in stars]
    total = max(sum(counts), 1)
    bars = ax.bar([str(s) for s in stars], counts, color="#2563eb", edgecolor="#1e3a8a", width=0.6)
    ax.set_title("MovieLens-1M Recorded Rating Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Interactions")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + total * 0.006,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(counts) * 1.14)
    fig.tight_layout()
    fig.savefig(ratings_fig_path)
    plt.close(fig)

    # 2. User sequence length histogram + L=100 cutoff marker
    lengths_fig_path = out_path / "ml1m_sequence_lengths.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    clipped = np.minimum(lengths, 600)
    ax.hist(clipped, bins=45, color="#0f766e", edgecolor="white", alpha=0.9)
    median_len = float(np.median(lengths))
    ax.axvline(median_len, color="#d97706", linestyle="--", linewidth=1.8, label=f"Median = {median_len:.0f}")
    ax.axvline(100, color="#dc2626", linestyle="-.", linewidth=1.8, label="SASRec L = 100")
    ax.set_title("MovieLens-1M User Interaction Sequence Lengths (Clipped at 600)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Interactions per User")
    ax.set_ylabel("Number of Users")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(lengths_fig_path)
    plt.close(fig)

    return {
        "rating_distribution": str(ratings_fig_path),
        "sequence_lengths": str(lengths_fig_path),
    }
