import React from "react";

const PRESETS = [
  {
    name: "Sci-Fi Classics",
    movies: [
      { item_id: 254, title: "Star Wars: Episode IV - A New Hope (1977)", genres: ["Action", "Adventure", "Fantasy", "Sci-Fi"] },
      { item_id: 528, title: "Blade Runner (1982)", genres: ["Film-Noir", "Sci-Fi"] },
      { item_id: 576, title: "Terminator 2: Judgment Day (1991)", genres: ["Action", "Sci-Fi", "Thriller"] },
      { item_id: 2709, title: "Total Recall (1990)", genres: ["Action", "Adventure", "Sci-Fi", "Thriller"] },
      { item_id: 2375, title: "Matrix, The (1999)", genres: ["Action", "Sci-Fi", "Thriller"] },
    ],
  },
  {
    name: "90s Crime & Noir",
    movies: [
      { item_id: 288, title: "Pulp Fiction (1994)", genres: ["Crime", "Drama"] },
      { item_id: 1124, title: "GoodFellas (1990)", genres: ["Crime", "Drama"] },
      { item_id: 594, title: "Fargo (1996)", genres: ["Crime", "Drama", "Thriller"] },
      { item_id: 50, title: "Usual Suspects, The (1995)", genres: ["Crime", "Thriller"] },
      { item_id: 159, title: "Devil in a Blue Dress (1995)", genres: ["Crime", "Film-Noir", "Mystery", "Thriller"] },
      { item_id: 1092, title: "Grifters, The (1990)", genres: ["Crime", "Drama", "Film-Noir"] },
    ],
  },
  {
    name: "Animation & Family",
    movies: [
      { item_id: 1, title: "Toy Story (1995)", genres: ["Animation", "Children's", "Comedy"] },
      { item_id: 575, title: "Aladdin (1992)", genres: ["Animation", "Children's", "Comedy", "Musical"] },
      { item_id: 355, title: "Lion King, The (1994)", genres: ["Animation", "Children's", "Musical"] },
      { item_id: 582, title: "Beauty and the Beast (1991)", genres: ["Animation", "Children's", "Musical"] },
    ],
  },
];

export default function SearchMoviesPanel({
  query,
  onQueryChange,
  searchResults,
  searching,
  onAddMovie,
  onApplyPreset,
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">1. Search MovieLens-1M Catalog &amp; Build Ordered History</h2>
        <span className="mono" style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
          Real titles from data/item2title.json (IDs 1..3706)
        </span>
      </div>

      <div className="search-row">
        <input
          type="text"
          className="search-input"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Search MovieLens-1M titles (e.g. Matrix, Star Wars, Toy Story, Godfather)..."
          aria-label="Search MovieLens-1M movies"
        />
        {searching && (
          <span className="mono" style={{ fontSize: "0.78rem", color: "#9ca3af" }}>
            Searching...
          </span>
        )}
      </div>

      <div className="preset-row">
        <span className="preset-label">Quick Presets:</span>
        {PRESETS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            className="btn btn-secondary"
            onClick={() => onApplyPreset(preset.movies)}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <div className="search-results">
        {searchResults.length === 0 ? (
          <div className="empty-state">No matching MovieLens-1M titles found.</div>
        ) : (
          searchResults.map((movie) => (
            <div key={movie.item_id} className="search-item">
              <div style={{ minWidth: 0 }}>
                <div className="search-item-title">{movie.title}</div>
                <div className="search-item-meta mono">
                  ID #{movie.item_id} &middot; {(movie.genres || []).join(" | ")}
                </div>
              </div>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onAddMovie(movie)}
              >
                + Add
              </button>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
