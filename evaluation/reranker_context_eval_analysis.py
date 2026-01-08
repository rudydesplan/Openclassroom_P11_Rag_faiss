import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "reranker_context_eval_results.csv"
TOP_K = 5

LOW_GAP_THRESHOLD = 0.01
HIGH_GAP_THRESHOLD = 0.05

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# Safety / typing
numeric_cols = [
    "context_position",
    "reranker_rank",
    "dense_rank",
    "delta_rank",
    "reranker_score_gap",
    "reranker_confidence_gap",
    "top_k",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 1️⃣ Core Context Builder KPIs
# ============================================================

context_recall_rate = df["context_recall"].mean()

mean_context_position = (
    df.loc[df.context_recall == 1, "context_position"].mean()
)

top1_hit_rate = (df["context_position"] == 1).mean()

print("\n=== Core Context Builder KPIs ===")
print(f"Context Recall Rate           : {context_recall_rate:.4f}")
print(f"Mean Context Position (recall): {mean_context_position:.4f}")
print(f"Top-1 Context Hit Rate        : {top1_hit_rate:.4f}")

# ============================================================
# 2️⃣ Context Risk & Failure Metrics
# ============================================================

dangerous_injection_rate = df["dangerous_context_injection"].mean()

reranker_disagreement_rate = (
    df["reranker_top_uid"] != df["ground_truth_uid"]
).mean()

print("\n=== Context Risk Metrics ===")
print(f"Dangerous Context Injection Rate : {dangerous_injection_rate:.4f}")
print(f"Reranker vs GT Disagreement Rate : {reranker_disagreement_rate:.4f}")

# ============================================================
# 3️⃣ Context Distributional Analysis
# ============================================================

position_distribution = (
    df["context_position"]
    .value_counts(normalize=True)
    .sort_index()
)

bucket_distribution = (
    df.assign(
        pos_bucket=pd.cut(
            df["context_position"],
            bins=[0, 1, 3, 5],
            labels=["1", "2-3", "4-5"],
            include_lowest=True
        )
    )
    .assign(
        pos_bucket=lambda x: x["pos_bucket"]
        .cat.add_categories(["missing"])
        .fillna("missing")
    )
    .pos_bucket.value_counts(normalize=True)
    .sort_index()
)

print("\n=== Context Position Distribution ===")
print(position_distribution)

print("\n=== Recall by Position Bucket ===")
print(bucket_distribution)

# ============================================================
# 4️⃣ Context Stability & Robustness
# ============================================================

per_uid_stats = (
    df.groupby("ground_truth_uid")["context_position"]
    .agg(["mean", "std"])
)

df["q_len"] = df["query"].astype(str).str.len()

question_length_sensitivity = (
    df.groupby(
        pd.qcut(df["q_len"], 4),
        observed=True
    )["context_position"]
    .mean()
)

print("\n=== Consistency per UID (mean / std) ===")
print(per_uid_stats.describe())

print("\n=== Question Length Sensitivity ===")
print(question_length_sensitivity)

# ============================================================
# 5️⃣ Advanced Context Metrics
# ============================================================

df["expected_context_cost"] = (df["context_position"] - 1) / df["top_k"]
expected_cost_mean = df["expected_context_cost"].mean()

llm_attention_risk = (
    (df.context_position > 2) |
    (df.dangerous_context_injection == 1)
).mean()

def context_ndcg(pos):
    if pd.isna(pos):
        return 0.0
    return 1 / np.log2(int(pos) + 1)

df["context_ndcg"] = df["context_position"].apply(context_ndcg)
mean_context_ndcg = df["context_ndcg"].mean()

print("\n=== Advanced Context Metrics ===")
print(f"Expected Context Cost    : {expected_cost_mean:.4f}")
print(f"LLM Attention Risk Index : {llm_attention_risk:.4f}")
print(f"Mean Context NDCG        : {mean_context_ndcg:.4f}")

# ============================================================
# 6️⃣ Reranker Rank Analysis
# ============================================================

rank_df = df.dropna(subset=["reranker_rank"]).copy()

print("\n=== Reranker Rank Evaluation ===")
print(f"Samples          : {len(rank_df)}")
print(f"Mean rank        : {rank_df['reranker_rank'].mean():.2f}")
print(f"Median (P50)     : {rank_df['reranker_rank'].median():.0f}")
print(f"P90 rank         : {rank_df['reranker_rank'].quantile(0.90):.0f}")
print(f"P95 rank         : {rank_df['reranker_rank'].quantile(0.95):.0f}")

print("\n--- Rank distribution (%) ---")
rank_dist = (
    rank_df["reranker_rank"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .reindex(range(1, TOP_K + 1), fill_value=0.0)
)
print(rank_dist)

# ============================================================
# 7️⃣ Delta Rank Analysis
# ============================================================

delta_df = df.dropna(subset=["delta_rank"]).copy()

improved = (delta_df["delta_rank"] > 0).mean() * 100
neutral = (delta_df["delta_rank"] == 0).mean() * 100
degraded = (delta_df["delta_rank"] < 0).mean() * 100

print("\n=== Reranker Delta Rank Evaluation ===")
print(f"Samples           : {len(delta_df)}")
print(f"Mean Δ rank       : {delta_df['delta_rank'].mean():.2f}")
print(f"Median (P50)      : {delta_df['delta_rank'].median():.0f}")
print(f"P95 Δ             : {delta_df['delta_rank'].quantile(0.95):.0f}")
print(f"Improved (Δ>0)    : {improved:.1f}%")
print(f"No change (Δ=0)   : {neutral:.1f}%")
print(f"Degraded (Δ<0)    : {degraded:.1f}%")

# ============================================================
# 8️⃣ Reranker Score Gap Analysis
# ============================================================

gap_df = df.dropna(subset=["reranker_score_gap"]).copy()

print("\n=== Reranker Score Gap Evaluation ===")
print(f"Samples           : {len(gap_df)}")
print(f"Mean gap          : {gap_df['reranker_score_gap'].mean():.4f}")
print(f"Median (P50)      : {gap_df['reranker_score_gap'].median():.4f}")
print(f"P95 gap           : {gap_df['reranker_score_gap'].quantile(0.95):.4f}")
print(f"P99 gap           : {gap_df['reranker_score_gap'].quantile(0.99):.4f}")
print(f"Max gap           : {gap_df['reranker_score_gap'].max():.4f}")

print("\n--- Confidence buckets (%) ---")
print(
    f"Low gap (≤ {LOW_GAP_THRESHOLD}) : "
    f"{(gap_df['reranker_score_gap'] <= LOW_GAP_THRESHOLD).mean() * 100:.1f}%"
)
print(
    f"High gap (≥ {HIGH_GAP_THRESHOLD}): "
    f"{(gap_df['reranker_score_gap'] >= HIGH_GAP_THRESHOLD).mean() * 100:.1f}%"
)

# ============================================================
# 9️⃣ Delta Rank vs Score Gap Correlation
# ============================================================

corr_df = df.dropna(subset=["delta_rank", "reranker_score_gap"]).copy()

pearson = corr_df["delta_rank"].corr(
    corr_df["reranker_score_gap"], method="pearson"
)
spearman = corr_df["delta_rank"].corr(
    corr_df["reranker_score_gap"], method="spearman"
)

print("\n=== Delta Rank vs Score Gap Correlation ===")
print(f"Pearson  : {pearson:.3f}")
print(f"Spearman : {spearman:.3f}")

print("\n--- Score gap by delta rank category ---")
for label, subset in {
    "Improved (Δ>0)": corr_df[corr_df["delta_rank"] > 0],
    "No change (Δ=0)": corr_df[corr_df["delta_rank"] == 0],
    "Degraded (Δ<0)": corr_df[corr_df["delta_rank"] < 0],
}.items():
    if len(subset) == 0:
        continue
    print(
        f"{label:<18} | "
        f"count={len(subset):5d} | "
        f"mean gap={subset['reranker_score_gap'].mean():.3f} | "
        f"P95 gap={subset['reranker_score_gap'].quantile(0.95):.3f}"
    )

# ============================================================
# 🔟 Confidence Gating Effect
# ============================================================

if "reranker_gated" in df.columns:
    print("\n=== Confidence Gating Analysis ===")
    print(f"Gated queries (%) : {df['reranker_gated'].mean() * 100:.2f}%")

    print("\nContext NDCG by gating:")
    print(df.groupby("reranker_gated")["context_ndcg"].mean())

    print("\nDangerous injection by gating:")
    print(df.groupby("reranker_gated")["dangerous_context_injection"].mean())


# -------------------------------------------------
# Dangerous cases: degraded + high confidence
# -------------------------------------------------
HIGH_GAP_THRESHOLD = 0.05

dangerous = df[
    (df["delta_rank"] < 0) & (df["reranker_score_gap"] >= HIGH_GAP_THRESHOLD)
]

print("\n=== Dangerous reranker errors ===")
print(f"Count            : {len(dangerous)}")
print(
    f"Percentage       : {100 * len(dangerous) / len(df):.2f}%"
)
print(
    f"Mean score gap   : {dangerous['reranker_score_gap'].mean():.3f}"
)



# ============================================================
# 11️⃣ One-line Summary
# ============================================================

print("\n=== FINAL SUMMARY ===")
print(
    f"Context Recall={context_recall_rate:.3f} | "
    f"Top1={top1_hit_rate:.3f} | "
    f"MeanPos={mean_context_position:.2f} | "
    f"NDCG={mean_context_ndcg:.3f} | "
    f"Danger={dangerous_injection_rate:.3f} | "
    f"Gated={df['reranker_gated'].mean() if 'reranker_gated' in df.columns else 0:.3f}"
)
