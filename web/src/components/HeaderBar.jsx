import React from "react";

export default function HeaderBar({ health, latencyMs }) {
  const isReal = health?.mode === "real";

  return (
    <header className="header-bar">
      <div>
        <h1 className="header-title">
          Team 39 — MovieLens-1M Sequential Recommender Studio
        </h1>
        <p className="header-sub">
          Pure-PyTorch SASRec (Causal Self-Attention) vs. GRU4Rec vs. Most-Popular
          &middot; Full-Catalog Ranking (N = {health?.catalog_size ?? 3706})
        </p>
      </div>

      <div className="telemetry-strip">
        <span className={`pill mono ${isReal ? "pill-real" : "pill-mock"}`}>
          ● {isReal ? "Real PyTorch Checkpoints" : "Mock Mode"}
        </span>
        {health?.mapping_version && (
          <span className="pill mono">{health.mapping_version}</span>
        )}
        {health?.device && (
          <span className="pill mono">Device: {health.device}</span>
        )}
        <span className="pill mono pill-latency">
          {latencyMs !== null ? `Measured Latency: ${latencyMs.toFixed(1)} ms` : "Ready"}
        </span>
      </div>
    </header>
  );
}
