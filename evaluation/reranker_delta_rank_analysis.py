import pandas as pd

# -------------------------------------------------
# Config
# -------------------------------------------------
CSV_PATH = "reranker_delta_rank.csv"

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Keep only delta rank metric
df = df[df["name"] == "reranker_delta_rank"].copy()

# Cast to numeric
df["delta_rank"] = pd.to_numeric(df["value"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["delta_rank"])

# -------------------------------------------------
# Global statistics
# -------------------------------------------------
stats = {
    "count": int(df.shape[0]),
    "mean_delta": df["delta_rank"].mean(),
    "median_p50": df["delta_rank"].median(),
    "p90": df["delta_rank"].quantile(0.90),
    "p95": df["delta_rank"].quantile(0.95),
    "min": df["delta_rank"].min(),
    "max": df["delta_rank"].max(),
}

# -------------------------------------------------
# Improvement / Neutral / Degradation split
# -------------------------------------------------
improved = (df["delta_rank"] > 0).mean() * 100
neutral = (df["delta_rank"] == 0).mean() * 100
degraded = (df["delta_rank"] < 0).mean() * 100

# -------------------------------------------------
# Pretty output
# -------------------------------------------------
print("\n=== Reranker Delta Rank Evaluation ===\n")

print(f"Samples (count)  : {stats['count']}")
print(f"Mean delta rank  : {stats['mean_delta']:.2f}")
print(f"Median (P50)     : {stats['median_p50']:.0f}")
print(f"P90 delta        : {stats['p90']:.0f}")
print(f"P95 delta        : {stats['p95']:.0f}")
print(f"Min delta        : {stats['min']:.0f}")
print(f"Max delta        : {stats['max']:.0f}")

print("\n--- Effect distribution (%) ---")
print(f"Improved (Δ > 0) : {improved:.1f}%")
print(f"No change (Δ=0)  : {neutral:.1f}%")
print(f"Degraded (Δ < 0) : {degraded:.1f}%")

# -------------------------------------------------
# Optional: save outputs
# -------------------------------------------------
# pd.DataFrame([stats]).to_csv("reranker_delta_rank_stats.csv", index=False)
