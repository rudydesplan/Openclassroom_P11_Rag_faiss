import pandas as pd

# -------------------------------------------------
# Config
# -------------------------------------------------
CSV_PATH = "reranker_score_gap.csv"

# Thresholds to interpret confidence
LOW_GAP_THRESHOLD = 0.01
HIGH_GAP_THRESHOLD = 0.05

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Keep only reranker score gap metric
df = df[df["name"] == "reranker_score_gap"].copy()

# Cast to numeric
df["score_gap"] = pd.to_numeric(df["value"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["score_gap"])

# -------------------------------------------------
# Global statistics
# -------------------------------------------------
stats = {
    "count": int(df.shape[0]),
    "mean_gap": df["score_gap"].mean(),
    "median_p50": df["score_gap"].median(),
    "p95": df["score_gap"].quantile(0.95),
    "p99": df["score_gap"].quantile(0.99),
    "max": df["score_gap"].max(),
}

# -------------------------------------------------
# Confidence interpretation
# -------------------------------------------------
near_zero = (df["score_gap"] <= LOW_GAP_THRESHOLD).mean() * 100
high_confidence_error = (df["score_gap"] >= HIGH_GAP_THRESHOLD).mean() * 100

# -------------------------------------------------
# Pretty output
# -------------------------------------------------
print("\n=== Reranker Score Gap Evaluation ===\n")

print(f"Samples (count) : {stats['count']}")
print(f"Mean gap        : {stats['mean_gap']:.4f}")
print(f"Median (P50)    : {stats['median_p50']:.4f}")
print(f"P95 gap         : {stats['p95']:.4f}")
print(f"P99 gap         : {stats['p99']:.4f}")
print(f"Max gap         : {stats['max']:.4f}")

print("\n--- Confidence analysis (%) ---")
print(f"Low gap (≤ {LOW_GAP_THRESHOLD}) : {near_zero:.1f}%")
print(f"High gap (≥ {HIGH_GAP_THRESHOLD}): {high_confidence_error:.1f}%")

# -------------------------------------------------
# Optional: save outputs
# -------------------------------------------------
# pd.DataFrame([stats]).to_csv("reranker_score_gap_stats.csv", index=False)
