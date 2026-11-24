import React, { useEffect, useState } from "react";
import HeaderBar from "./components/HeaderBar.jsx";
import SearchMoviesPanel from "./components/SearchMoviesPanel.jsx";
import SequenceTrack from "./components/SequenceTrack.jsx";
import RecommendationColumn from "./components/RecommendationColumn.jsx";
import AttentionPanel from "./components/AttentionPanel.jsx";

const INITIAL_HISTORY = [
  { item_id: 254, title: "Star Wars: Episode IV - A New Hope (1977)", genres: ["Action", "Adventure", "Fantasy", "Sci-Fi"] },
  { item_id: 528, title: "Blade Runner (1982)", genres: ["Film-Noir", "Sci-Fi"] },
  { item_id: 576, title: "Terminator 2: Judgment Day (1991)", genres: ["Action", "Sci-Fi", "Thriller"] },
  { item_id: 2709, title: "Total Recall (1990)", genres: ["Action", "Adventure", "Sci-Fi", "Thriller"] },
  { item_id: 2375, title: "Matrix, The (1999)", genres: ["Action", "Sci-Fi", "Thriller"] },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [history, setHistory] = useState(INITIAL_HISTORY);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [latencyMs, setLatencyMs] = useState(null);

  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch(() => setError("Cannot connect to FastAPI backend at /api/health."));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setSearching(true);
    const timer = setTimeout(() => {
      fetch(`/api/movies?q=${encodeURIComponent(query)}&limit=8`, {
        signal: controller.signal,
      })
        .then((res) => res.json())
        .then((data) => setSearchResults(data.items || []))
        .catch((err) => {
          if (err.name !== "AbortError") {
            setError(`Movie catalog search failed: ${err.message}`);
          }
        })
        .finally(() => setSearching(false));
    }, 150);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  const handleAddMovie = (movie) => {
    setError("");
    setHistory((prev) => [...prev, movie]);
  };

  const handleMove = (index, delta) => {
    setHistory((prev) => {
      const nextIndex = index + delta;
      if (nextIndex < 0 || nextIndex >= prev.length) return prev;
      const updated = [...prev];
      const [moved] = updated.splice(index, 1);
      updated.splice(nextIndex, 0, moved);
      return updated;
    });
  };

  const handleRemove = (index) => {
    setHistory((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleRunRecommend = async (overrideHistory = null) => {
    const seq = Array.isArray(overrideHistory) ? overrideHistory : history;
    if (seq.length === 0) {
      setError("Please add at least one MovieLens-1M movie before requesting recommendations.");
      return;
    }

    setError("");
    setLoading(true);
    const started = performance.now();
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          history_ids: seq.map((m) => m.item_id),
          top_k: 10,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || `HTTP ${res.status}`);
      }
      setRecommendations(payload);
      setLatencyMs(performance.now() - started);
    } catch (err) {
      setError(`Recommendation request failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyPreset = (presetMovies) => {
    setHistory(presetMovies);
    handleRunRecommend(presetMovies);
  };

  useEffect(() => {
    handleRunRecommend(INITIAL_HISTORY);
  }, []);

  return (
    <main className="app-shell">
      <HeaderBar health={health} latencyMs={latencyMs} />

      {error && <div className="error-banner">{error}</div>}

      <SearchMoviesPanel
        query={query}
        onQueryChange={setQuery}
        searchResults={searchResults}
        searching={searching}
        onAddMovie={handleAddMovie}
        onApplyPreset={handleApplyPreset}
      />

      <SequenceTrack
        history={history}
        loading={loading}
        onMove={handleMove}
        onRemove={handleRemove}
        onClear={() => {
          setHistory([]);
          setRecommendations(null);
        }}
        onRunRecommend={() => handleRunRecommend()}
      />

      <section className="compare-grid">
        <RecommendationColumn
          variant="sasrec"
          title="SASRec"
          badgeText="Causal Self-Attention"
          subtitle="Pure-PyTorch Transformer (d=64, 2 blocks, 2 heads, right-padding L=100)"
          scoreUnit="Raw Logit"
          items={recommendations?.sasrec_recommendations}
        />
        <RecommendationColumn
          variant="gru"
          title="GRU4Rec"
          badgeText="Recurrent Baseline"
          subtitle="Pure-PyTorch GRU (d=64, 1 layer, right-padding L=100)"
          scoreUnit="Raw Logit"
          items={recommendations?.gru4rec_recommendations}
        />
        <RecommendationColumn
          variant="pop"
          title="Most-Popular"
          badgeText="Train-Only Baseline"
          subtitle="Non-sequential global frequency counts computed strictly from Train split"
          scoreUnit="Raw Count"
          items={recommendations?.popular_recommendations}
        />
      </section>

      <AttentionPanel attention={recommendations?.attention} history={history} />
    </main>
  );
}
