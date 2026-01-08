import pandas as pd

# -------------------------------------------------
# Config
# -------------------------------------------------
CSV_PATH = "retriever_score_gap.csv"

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Sécurité : garder uniquement le score retriever_score_gap
df = df[df["name"] == "retriever_score_gap"].copy()

# Cast en numérique
df["score_gap"] = pd.to_numeric(df["value"], errors="coerce")

# Supprimer lignes invalides
df = df.dropna(subset=["score_gap"])

# (Optionnel) sécurité : score_gap doit être >= 0
df = df[df["score_gap"] >= 0]

# -------------------------------------------------
# Global statistics
# -------------------------------------------------
stats = {
    "mean_gap": df["score_gap"].mean(),
    "median_gap": df["score_gap"].median(),
    "p95_gap": df["score_gap"].quantile(0.95),
    "p99_gap": df["score_gap"].quantile(0.99),
    "count": int(df.shape[0]),
}

# -------------------------------------------------
# Pretty output (console)
# -------------------------------------------------
print("\n=== Retriever Score Gap Evaluation ===\n")

print(f"Samples (count) : {stats['count']}")
print(f"Mean gap        : {stats['mean_gap']:.4f}")
print(f"Median gap      : {stats['median_gap']:.4f}")
print(f"P95 gap         : {stats['p95_gap']:.4f}")
print(f"P99 gap         : {stats['p99_gap']:.4f}")

# -------------------------------------------------
# Save outputs
# -------------------------------------------------
#pd.DataFrame([stats]).to_csv("retriever_score_gap_stats.csv", index=False)

#print("\nSaved file:")
#print("- retriever_score_gap_stats.csv")
