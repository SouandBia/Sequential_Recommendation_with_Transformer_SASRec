"""Plot a sequence-level SASRec causal attention weight heatmap diagnostic."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _shorten_title(title: str, max_len: int = 24) -> str:
    return title if len(title) <= max_len else title[: max_len - 3] + "..."


def plot_attention_heatmap(
    attention_matrix: list[list[float]],
    item_titles: list[str],
    layer_idx: int = 1,
    head_idx: int = 0,
    max_display_steps: int = 12,
    output_paths: list[str | Path] | None = None,
) -> list[str]:
    """Render a causal attention heatmap over the most recent steps with diagnostic caveat."""
    mat = np.asarray(attention_matrix, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        raise ValueError(f"attention_matrix must be non-empty square 2D, got {mat.shape}")
    if len(item_titles) != mat.shape[0]:
        raise ValueError("item_titles length must match attention_matrix dimension.")

    targets = [Path(p) for p in (output_paths or ["results/figures/attention_heatmap.png"])]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)

    k = min(max_display_steps, mat.shape[0])
    sub_mat = mat[-k:, -k:]
    labels = [f"t-{k - 1 - idx}: {_shorten_title(t)}" for idx, t in enumerate(item_titles[-k:])]

    fig, ax = plt.subplots(figsize=(8.4, 7.0), dpi=160)
    im = ax.imshow(sub_mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=float(max(sub_mat.max(), 1e-4)))

    ax.set_xticks(range(k), labels, rotation=42, ha="right", fontsize=8)
    ax.set_yticks(range(k), labels, fontsize=8)
    ax.set_xlabel("Key Position (Past Items Attended To)")
    ax.set_ylabel("Query Position (Sequence Step)")
    ax.set_title(
        f"SASRec Causal Self-Attention Diagnostic (Block {layer_idx + 1}, Head {head_idx + 1})\n"
        "Sequence-level position relationships only — not causal proof for a recommended movie",
        fontsize=10.2,
        fontweight="bold",
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=9)

    fig.tight_layout()
    saved: list[str] = []
    for path in targets:
        fig.savefig(path)
        saved.append(str(path))
    plt.close(fig)
    return saved
