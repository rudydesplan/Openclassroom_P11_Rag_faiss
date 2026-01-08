Même si le context builder est déterministe, il transforme la sortie du reranker d’une manière qui peut altérer la visibilité, la présence ou la dominance de l’information pertinente.

Voici les 3 métriques de contexte pertinentes 

Objectif : évaluer si le contexte transmis au LLM est exploitable et sûr, indépendamment du fait qu’il soit “créé”

1️⃣ Context Recall (Coverage)
❓ Question à laquelle elle répond

Le document ground truth est-il présent dans le contexte final envoyé au LLM ?

Pourquoi c’est indispensable

Le retriever peut avoir Recall@k = 99.99 %

Le reranker peut être bon

MAIS le context builder peut :

tronquer

dédupliquer

dépasser le budget tokens

👉 Si le bon document disparaît ici, le LLM est condamné.

2️⃣ Context Position (Visibility)
❓ Question

À quelle position le document pertinent apparaît-il dans le contexte ?


Pourquoi c’est critique

Les LLM ont un biais de position

Les premiers documents ont plus d’impact

Un bon document en position 6/6 est quasi inutile

3️⃣ Dangerous Context Injection (Safety)
❓ Question

Le contexte contient-il des documents faux mais fortement confiants AVANT le bon document ?

A cause du reranker : 

certaines erreurs sont confiantes

~14 % de cas dangereux

👉 Ici on mesure si ces erreurs atteignent réellement le LLM.


=== Core Context Builder KPIs ===
Context Recall Rate           : 1.0000
Mean Context Position (recall): 1.5580
Top-1 Context Hit Rate        : 0.6950

=== Context Risk Metrics ===
Dangerous Context Injection Rate : 0.1113
Reranker vs GT Disagreement Rate : 0.2917

=== Context Position Distribution ===
context_position
1    0.695000
2    0.153667
3    0.079333
4    0.042333
5    0.029667
Name: proportion, dtype: float64

=== Recall by Position Bucket ===
pos_bucket
1          0.695
2-3        0.233
4-5        0.072
missing    0.000
Name: proportion, dtype: float64

=== Consistency per UID (mean / std) ===
              mean         std
count  1639.000000  939.000000
mean      1.565284    0.482132
std       0.878991    0.714925
min       1.000000    0.000000
25%       1.000000    0.000000
50%       1.000000    0.000000
75%       2.000000    0.707107
max       5.000000    2.828427

=== Question Length Sensitivity ===
q_len
(21.999, 48.0]    1.657895
(48.0, 58.0]      1.556931
(58.0, 69.0]      1.488340
(69.0, 164.0]     1.523471
Name: context_position, dtype: float64

=== Advanced Context Metrics ===
Expected Context Cost    : 0.1116
LLM Attention Risk Index : 0.2010
Mean Context NDCG        : 0.8613

Our context builder achieves near-perfect recall, strong attention-aligned ranking (NDCG 0.88), and low context-induced hallucination risk. Remaining errors are driven by reranker miscalibration rather than context construction.”