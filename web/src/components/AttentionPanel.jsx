import React, { useState } from "react";

export default function AttentionPanel({ attention, history }) {
  const [expanded, setExpanded] = useState(true);

  if (!attention || !Array.isArray(attention.weights) || attention.weights.length === 0) {
    return null;
  }

  const recentHistory = history.slice(-attention.weights.length);
  const maxWeight = Math.max(...attention.weights, 1e-6);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">
          Optional Diagnostic: SASRec Causal Self-Attention Weights (Block {attention.layer + 1}, Head {attention.head + 1})
        </h2>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      </div>

      {expanded && (
        <div>
          {attention.weights.map((weight, idx) => {
            const movie = recentHistory[idx];
            const barWidthPct = Math.min(100, (weight / maxWeight) * 100);
            return (
              <div key={`${idx}-${movie?.item_id ?? idx}`} className="attn-bar-row">
                <div
                  style={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: "#e5e7eb",
                  }}
                  title={movie?.title}
                >
                  <span className="mono" style={{ color: "#818cf8", marginRight: "0.4rem" }}>
                    t{idx + 1}
                  </span>
                  {movie?.title ?? `Item #${idx + 1}`}
                </div>
                <div className="attn-bar-track">
                  <div className="attn-bar-fill" style={{ width: `${barWidthPct}%` }} />
                </div>
                <div className="mono" style={{ textAlign: "right", color: "#c7c4d7" }}>
                  {Number(weight).toFixed(4)}
                </div>
              </div>
            );
          })}

          <div className="disclaimer-box">
            <strong>Diagnostic Notice:</strong> Attention weights show sequence-level
            position-to-position weighting at the last valid query position inside Block{" "}
            {attention.layer + 1}, Head {attention.head + 1}. They do{" "}
            <strong>not</strong> constitute causal proof explaining why a specific movie was recommended.
          </div>
        </div>
      )}
    </section>
  );
}
