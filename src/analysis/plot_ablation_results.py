"""Plot post-MVP SASRec sequence-length ablation results (L in {50, 100, 200})."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ablation_results(
    ablation_rows: list[dict],
    output_paths: list[str | Path] | None = None,
) -> list[str]:
    """Plot validation NDCG@10, HR@10, and peak RTX 4060 VRAM across max_seq_len values."""
    if not ablation_rows:
        raise ValueError("ablation_rows must not be empty.")

    targets = [Path(p) for p in (output_paths or ["results/figures/ablation_seq_len.png"])]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(ablation_rows, key=lambda r: int(r["max_seq_len"]))
    seq_lens = [str(int(r["max_seq_len"])) for r in ordered]
    ndcgs = [float(r["val_NDCG@10"]) for r in ordered]
    hrs = [float(r["val_HR@10"]) for r in ordered]
    vrams = [float(r["peak_vram_mb"]) for r in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)

    x_pos = list(range(len(seq_lens)))
    width = 0.34
    bars_hr = axes[0].bar([x - width / 2 for x in x_pos], hrs, width=width, color="#2563eb", label="Val HR@10")
    bars_ndcg = axes[0].bar([x + width / 2 for x in x_pos], ndcgs, width=width, color="#0f766e", label="Val NDCG@10")
    axes[0].set_xticks(x_pos, [f"L={s}" for s in seq_lens])
    axes[0].set_title("SASRec Validation Accuracy vs Max Sequence Length", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Full-Catalog Metric")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True)
    axes[0].set_ylim(0, max(hrs + ndcgs) * 1.22)
    for bar in list(bars_hr) + list(bars_ndcg):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    bars_vram = axes[1].bar(x_pos, vrams, width=0.45, color="#d97706", edgecolor="#92400e")
    axes[1].set_xticks(x_pos, [f"L={s}" for s in seq_lens])
    axes[1].set_title("SASRec Peak Training VRAM vs Max Sequence Length", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Peak VRAM (MB)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].set_ylim(0, max(vrams) * 1.25 if max(vrams) > 0 else 1.0)
    for bar in bars_vram:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, h + max(vrams) * 0.02, f"{h:.0f} MB", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    saved: list[str] = []
    for path in targets:
        fig.savefig(path)
        saved.append(str(path))
    plt.close(fig)
    return saved
