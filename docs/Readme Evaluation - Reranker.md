Complementary reranker metrics:

Metric	What it measures

Reranker Rank	: How early the reranker positions the correct document after reordering

Reranker Score Gap	 :  How strongly the reranker prefers an incorrect document over the correct one

Delta Rank	:  How much the reranker improves (or degrades) the document position compared to the dense retriever

# 🔍 Évaluation du Reranker et Stratégie de Gating

## 1. Performance du retriever

Le retriever dense atteint un taux de rappel de 99,99 %, garantissant que le document pertinent est presque toujours présent dans le top-K candidats transmis au reranker.
L’enjeu principal se situe donc au niveau du classement et de la sélection du contexte, et non de la récupération.

---

## 2. Performance intrinsèque du reranker

### 2.1 Qualité de classement

Sur 3 000 requêtes évaluées :

 Rang moyen : 1,50
 Médiane (P50) : 1
 P90 : 3
 P95 : 4

Distribution des rangs :

 Rang 1 : 70,8 %
 Rang ≤ 3 : 94,3 %

Ces résultats montrent que le reranker améliore effectivement le positionnement du document pertinent dans environ 18 % des cas, tout en laissant le classement inchangé dans 65 % des requêtes.

---

### 2.2 Impact du reranker par rapport au retriever

 Amélioration du rang (Δ > 0) : 18,2 %
 Aucun impact (Δ = 0) : 65,4 %
 Dégradation du rang (Δ < 0) : 16,3 %

Ainsi, bien que le reranker apporte un gain mesurable, il introduit également un risque non négligeable de dégradation du classement.

---

## 3. Analyse de la confiance du reranker

### 3.1 Distribution du score de confiance

 Score moyen : 0,32
 Médiane : 0,00
 P95 : 2,11
 P99 : 4,01

Répartition :

 Faible confiance (≤ 0,01) : 71,9 %
 Forte confiance (≥ 0,05) : 26,1 %

---

### 3.2 Corrélation confiance ↔ erreur

L’analyse révèle une corrélation négative significative entre le score de confiance et la variation de rang :

 Pearson : –0,308
 Spearman : –0,420

En particulier :

 Les dégradations de rang sont associées à un score moyen de 1,23
 Les améliorations présentent un score moyen bien plus faible (0,27)

👉 Les erreurs du reranker ne sont donc pas aléatoires, mais souvent confiantes.

---

## 4. Risque d’hallucination et erreurs dangereuses

Parmi l’ensemble des requêtes :

 15,03 % correspondent à des erreurs dangereuses
 Score de confiance moyen de ces erreurs : 1,33

Ces cas représentent un risque direct d’hallucination, car le modèle de génération reçoit en priorité un contexte incorrect mais fortement mis en avant.

---

## 5. Introduction d’un mécanisme de confidence gating

### 5.1 Principe

Afin de rendre le pipeline exploitable en production (sans accès au ground truth), un mécanisme de confidence gating est introduit.
Le reranker est neutralisé uniquement lorsque :

 il est fortement confiant, et
 son classement entre en désaccord avec le retriever dense

---

### 5.2 Impact du gating

 Requêtes gated : 28,07 %
 Context NDCG sans gating : 0,926
 Context NDCG avec gating : 0,696

Surtout :

 Taux d’injection dangereuse (non-gated) : 4,77 %
 Taux d’injection dangereuse (gated) : 27,43 %

👉 Le mécanisme de gating concentre efficacement le risque sur un sous-ensemble réduit de requêtes, tout en laissant les requêtes sûres bénéficier pleinement du reranking.

---

## 6. Conclusion

Le reranker apporte un gain réel mais limité en termes de qualité de contexte (~18 % des cas).
Cependant, lorsqu’il se trompe, il le fait souvent avec une forte confiance, ce qui constitue un facteur critique de risque d’hallucination.

Par conséquent :

> Le reranker ne doit pas être utilisé seul, mais combiné à des mécanismes de contrôle basés sur la confiance.

L’introduction d’un confidence gate, dérivé d’une analyse offline et exploitable en production sans supervision, permet de préserver les bénéfices du reranking tout en améliorant la robustesse globale du pipeline RAG.

---

> The reranker provides measurable ranking gains in a minority of cases, but its confident errors pose a direct hallucination risk.
> Therefore, confidence-based gating is required to ensure robustness of the RAG pipeline in production.