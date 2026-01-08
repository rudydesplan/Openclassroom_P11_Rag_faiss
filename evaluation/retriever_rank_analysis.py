import pandas as pd

# -------------------------------------------------
# Config
# -------------------------------------------------

#CSV_PATH = "retriever_rank.csv"
CSV_PATH = "reranker_rank.csv"
K_MAX = 5                    

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Sécurité : garder uniquement le score retriever_rank
#df = df[df["name"] == "retriever_rank"].copy()

df = df[df["name"] == "reranker_rank"].copy() 

# Cast en numérique
df["rank"] = pd.to_numeric(df["value"], errors="coerce")

# Supprimer lignes invalides
df = df.dropna(subset=["rank"])

# (Optionnel) s'assurer que les rangs sont bien dans [1, K_MAX]
df = df[(df["rank"] >= 1) & (df["rank"] <= K_MAX)]

# -------------------------------------------------
# Global statistics
# -------------------------------------------------
stats = {
    "mean_rank": df["rank"].mean(),
    "median_rank_p50": df["rank"].median(),
    "p90_rank": df["rank"].quantile(0.90),
    "p95_rank": df["rank"].quantile(0.95),
    "count": int(df.shape[0]),
}

# -------------------------------------------------
# Discrete distribution (%) for ranks 1..K_MAX
# -------------------------------------------------
rank_distribution = (
    df["rank"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .reindex(range(1, K_MAX + 1), fill_value=0.0)
    .rename("percentage")
    .reset_index()
    .rename(columns={"index": "rank"})
)

# -------------------------------------------------
# Pretty output (console)
# -------------------------------------------------
#print("\n=== Retriever Rank Evaluation (k = {}) ===\n".format(K_MAX))
print("\n=== Reranker Rank Evaluation (k = {}) ===\n".format(K_MAX))

print(f"Samples (count)  : {stats['count']}")
print(f"Mean rank        : {stats['mean_rank']:.2f}")
print(f"Median (P50)     : {stats['median_rank_p50']:.0f}")
print(f"P90 rank         : {stats['p90_rank']:.0f}")
print(f"P95 rank         : {stats['p95_rank']:.0f}")

print("\n--- Rank distribution (%) ---")
for _, row in rank_distribution.iterrows():
    print(f"Rank {int(row['rank'])}: {row['percentage']:.1f}%")

# -------------------------------------------------
# Save outputs
# -------------------------------------------------
#rank_distribution.to_csv("retriever_rank_distribution.csv", index=False)
#pd.DataFrame([stats]).to_csv("retriever_rank_stats.csv", index=False)

#print("\nSaved files:")
#print("- retriever_rank_distribution.csv")
#print("- retriever_rank_stats.csv")
