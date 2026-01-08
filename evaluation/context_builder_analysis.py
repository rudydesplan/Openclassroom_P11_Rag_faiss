import pandas as pd
import numpy as np

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("context_eval_results.csv")

print(f"Loaded {len(df)} rows")

# Safety
df["context_position"] = pd.to_numeric(df["context_position"], errors="coerce")

# ============================================================
# 1️⃣ Core Context Builder KPIs
# ============================================================

context_recall_rate = df["context_recall"].mean()

mean_context_position = (
    df.loc[df.context_recall == 1, "context_position"].mean()
)

top1_hit_rate = (df["context_position"] == 1).mean()

print("\n=== Core Context Builder KPIs ===")
print(f"Context Recall Rate: {context_recall_rate:.4f}")
print(f"Mean Context Position (recall=1): {mean_context_position:.4f}")
print(f"Top-1 Context Hit Rate: {top1_hit_rate:.4f}")

# ============================================================
# 2️⃣ Context Risk & Failure Metrics
# ============================================================

dangerous_injection_rate = df["dangerous_context_injection"].mean()

reranker_disagreement_rate = (
    df["reranker_top_uid"] != df["ground_truth_uid"]
).mean()

print("\n=== Context Risk Metrics ===")
print(f"Dangerous Context Injection Rate: {dangerous_injection_rate:.4f}")
print(f"Reranker vs GT Disagreement Rate: {reranker_disagreement_rate:.4f}")

# ============================================================
# 3️⃣ Distributional Analysis
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
        pos_bucket=lambda x: x["pos_bucket"].cat.add_categories(["missing"])
        .fillna({"pos_bucket": "missing"})
    )
    .pos_bucket.value_counts(normalize=True)
    .sort_index()
)

print("\n=== Context Position Distribution ===")
print(position_distribution)

print("\n=== Recall by Position Bucket ===")
print(bucket_distribution)

# ============================================================
# 4️⃣ Stability & Robustness Metrics
# ============================================================

per_uid_stats = (
    df.groupby("ground_truth_uid")["context_position"]
    .agg(["mean", "std"])
)

df["q_len"] = df["query"].str.len()

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
# 5️⃣ Advanced Metrics
# ============================================================

df["expected_context_cost"] = (df["context_position"] - 1) / df["top_k"]
expected_cost_mean = df["expected_context_cost"].mean()

llm_attention_risk = (
    (df.context_position > 2) |
    (df.dangerous_context_injection == 1)
).mean()

# ---- Context NDCG ----
def context_ndcg(pos):
    if np.isnan(pos):
        return 0.0
    return 1 / np.log2(int(pos) + 1)

df["context_ndcg"] = df["context_position"].apply(context_ndcg)
mean_context_ndcg = df["context_ndcg"].mean()

print("\n=== Advanced Metrics ===")
print(f"Expected Context Cost: {expected_cost_mean:.4f}")
print(f"LLM Attention Risk Index: {llm_attention_risk:.4f}")
print(f"Mean Context NDCG: {mean_context_ndcg:.4f}")

# ============================================================
# 6️⃣ Summary (single-line report)
# ============================================================

print("\n=== SUMMARY ===")
print(
    f"Recall={context_recall_rate:.3f} | "
    f"Top1={top1_hit_rate:.3f} | "
    f"MeanPos={mean_context_position:.2f} | "
    f"NDCG={mean_context_ndcg:.3f} | "
    f"Danger={dangerous_injection_rate:.3f}"
)
