import pandas as pd

# -------------------------------------------------
# Load data
# -------------------------------------------------
delta_df = pd.read_csv("reranker_delta_rank.csv")
gap_df = pd.read_csv("reranker_score_gap.csv")

delta_df = delta_df[delta_df["name"] == "reranker_delta_rank"].copy()
gap_df = gap_df[gap_df["name"] == "reranker_score_gap"].copy()

delta_df["delta_rank"] = pd.to_numeric(delta_df["value"], errors="coerce")
gap_df["score_gap"] = pd.to_numeric(gap_df["value"], errors="coerce")

# Align indices (important)
df = pd.concat(
    [
        delta_df["delta_rank"].reset_index(drop=True),
        gap_df["score_gap"].reset_index(drop=True),
    ],
    axis=1,
).dropna()

print(f"Samples aligned: {len(df)}")

# -------------------------------------------------
# Correlations
# -------------------------------------------------
pearson_corr = df["delta_rank"].corr(df["score_gap"], method="pearson")
spearman_corr = df["delta_rank"].corr(df["score_gap"], method="spearman")

print("\n=== Correlation analysis ===")
print(f"Pearson correlation : {pearson_corr:.3f}")
print(f"Spearman correlation: {spearman_corr:.3f}")

# -------------------------------------------------
# Score gap by delta category
# -------------------------------------------------
print("\n=== Score gap by delta rank category ===")

for label, subset in {
    "Improved (Δ > 0)": df[df["delta_rank"] > 0],
    "No change (Δ = 0)": df[df["delta_rank"] == 0],
    "Degraded (Δ < 0)": df[df["delta_rank"] < 0],
}.items():
    if len(subset) == 0:
        continue
    print(
        f"{label:<20} | "
        f"count={len(subset):4d} | "
        f"mean gap={subset['score_gap'].mean():.3f} | "
        f"P95 gap={subset['score_gap'].quantile(0.95):.3f}"
    )

# -------------------------------------------------
# Dangerous cases: degraded + high confidence
# -------------------------------------------------
HIGH_GAP_THRESHOLD = 0.05

dangerous = df[
    (df["delta_rank"] < 0) & (df["score_gap"] >= HIGH_GAP_THRESHOLD)
]

print("\n=== Dangerous reranker errors ===")
print(f"Count            : {len(dangerous)}")
print(
    f"Percentage       : {100 * len(dangerous) / len(df):.2f}%"
)
print(
    f"Mean score gap   : {dangerous['score_gap'].mean():.3f}"
)
