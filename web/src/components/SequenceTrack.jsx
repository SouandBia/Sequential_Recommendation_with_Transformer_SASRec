import React from "react";

export default function SequenceTrack({
  history,
  loading,
  onMove,
  onRemove,
  onClear,
  onRunRecommend,
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">
          2. Ordered Interaction Sequence (t<sub>1</sub> &rarr; t<sub>{history.length || "n"}</sub>)
        </h2>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
          <span className="pill mono">Length: {history.length} / L=100</span>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={onClear}
            disabled={loading || history.length === 0}
          >
            Clear Sequence
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onRunRecommend}
            disabled={loading}
          >
            {loading ? "Running Inference..." : "Run Top-10 Recommendation"}
          </button>
        </div>
      </div>

      <div className="sequence-track">
        {history.length === 0 ? (
          <div className="empty-state">
            Add at least one MovieLens-1M movie above (or pick a quick preset) to construct an ordered history.
          </div>
        ) : (
          history.map((movie, idx) => (
            <React.Fragment key={`${movie.item_id}-${idx}`}>
              <div
                className={`sequence-chip ${
                  idx === history.length - 1 ? "sequence-chip-latest" : ""
                }`}
              >
                <span className="seq-index mono">t{idx + 1}</span>
                <span className="seq-title">{movie.title}</span>
                <span className="mono" style={{ fontSize: "0.7rem", color: "#9ca3af" }}>
                  #{movie.item_id}
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  title="Move earlier"
                  disabled={idx === 0}
                  onClick={() => onMove(idx, -1)}
                >
                  &larr;
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  title="Move later"
                  disabled={idx === history.length - 1}
                  onClick={() => onMove(idx, 1)}
                >
                  &rarr;
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  title="Remove"
                  onClick={() => onRemove(idx)}
                >
                  &times;
                </button>
              </div>
              {idx < history.length - 1 && <span className="seq-arrow">&rarr;</span>}
            </React.Fragment>
          ))
        )}
      </div>
    </section>
  );
}
