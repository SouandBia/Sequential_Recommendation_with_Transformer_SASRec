import React from "react";

export default function RecommendationColumn({
  variant,
  title,
  badgeText,
  subtitle,
  scoreUnit,
  items,
}) {
  return (
    <div className={`model-col model-col-${variant}`}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.5rem" }}>
        <h3 style={{ margin: 0, fontSize: "1rem", color: "#f9fafb" }}>{title}</h3>
        <span className={`model-badge badge-${variant}`}>{badgeText}</span>
      </div>
      <p className="model-desc">{subtitle}</p>

      {!items || items.length === 0 ? (
        <div className="empty-state">Click &ldquo;Run Top-10 Recommendation&rdquo; to inspect predictions.</div>
      ) : (
        <ol className="rec-list">
          {items.map((item, idx) => (
            <li key={item.item_id} className="rec-card">
              <div className="rec-left">
                <span className="rec-rank mono">#{idx + 1}</span>
                <div style={{ minWidth: 0 }}>
                  <div className="rec-title">{item.title}</div>
                  <div className="genre-pills">
                    {(item.genres || []).map((genre) => (
                      <span key={genre} className="genre-pill">
                        {genre}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="raw-score-box">
                <span className="raw-score-label">{scoreUnit}</span>
                <span className="raw-score-val mono">
                  {variant === "pop"
                    ? Number(item.score).toFixed(0)
                    : Number(item.score).toFixed(4)}
                </span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
