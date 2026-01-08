3 complementary retriever metrics:

Metric	What it measures
Recall@k	Can the retriever find the correct document at all?
Rank	How early does it find it?
Score gap	How strongly does the retriever prefer the wrong doc over the correct one

Analytics : 
Retriever_recall_at_k = 99,99 % True

Rank :
=== Retriever Rank Evaluation (k = 5) ===

Samples (count)  : 9999
Mean rank        : 1.49
Median (P50)     : 1
P90 rank         : 3
P95 rank         : 4

--- Rank distribution (%) ---
Rank 1: 73.5%
Rank 2: 13.5%
Rank 3: 6.5%
Rank 4: 3.8%
Rank 5: 2.7%

Score gap :

Samples (count) : 10000
Mean gap        : 0.0053
Median gap      : 0.0000
P95 gap         : 0.0335
P99 gap         : 0.0687


The dense retriever (BGE-M3, k=5) shows strong performance on a 10k-query evaluation set, with a Recall@5 close to 1, a median rank of 1, and a mean rank of 1.49.
The score gap distribution is tightly concentrated near zero (P95 = 0.033), indicating that even failure cases exhibit low confidence.
These results suggest a robust and reliable retrieval stage suitable for downstream RAG generation without additional reranking.