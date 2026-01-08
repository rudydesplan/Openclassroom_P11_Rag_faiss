# 📘 **Étape 3 — Construction de l’Index Vectoriel & Pipeline RAG Avancé**

Cette étape constitue le cœur du pipeline RAG : la **transformation des documents en vecteurs**, leur **indexation**, puis la construction d’un **retriever dense**, d’un **reranker**, et d’un **context builder structuré**.

------

# 🎯 **Objectifs de l’Étape 3**

1. Transformer les documents nettoyés (Étape 2) en embeddings numériques.
2. Construire un index vectoriel performant (**FAISS CPU, IndexFlatIP**).
3. Charger et utiliser ce vecteur store via un **DenseRetriever**.
4. Ajouter un **reranker cross-encoder BAAI/bge-reranker-large**.
5. Construire un **Context Builder** optimisé pour le LLM.

------

# 🏗️ **Architecture de l’Étape 3**

📦 `outputs/documents_for_faiss.jsonl`
 ⬇️
 📘 **build_dense_index.py** → Encode + index FAISS
 ⬇️
 🧠 **dense_retriever.py** → Recherche dense top-k
 ⬇️
 🎯 **bge_reranker.py** → Reranking cross-encoder
 ⬇️
 📜 **context_builder.py** → Contexte final propre & structuré
 ⬇️
 🤖 Étape 4 → LLM (Gemma, Mistral, Llama, etc.)



------

# 1️⃣ **Construction de l’index vectoriel (build_dense_index.py)**

📄 Fichier utilisé : **build_dense_index.py** 

## ✔ Rôle

Encoder chaque document en un vecteur dense avec :

- **BAAI/bge-m3** → modèle multilingue, long-context, entraîné pour retrieval.
- **FAISS IndexFlatIP** → index simple, rapide, idéal pour prototypes et bases < 10M docs.

## ✔ Justification (livres)

### 🧠 *AI Engineering*, Chap. 6 (p. 267)

Huyen explique que :

- Il faut **privilégier les architectures simples** avant d’ajouter de la complexité.
- Les index *flat* CPU permettent une **reproductibilité parfaite** et des scores 1-to-1.
- La normalisation L2 + inner product = **cosine similarity**, recommandée pour BGE.

### 📘 *RAG-Driven Generative AI*, Chap. 2

Rothman recommande :

- le batching massif pour réduire la latence,
- l’indexation incrémentale,
- des pipelines stateless reproductibles.

## ✔ Comportement du script

- Lecture **streaming** JSONL → évite 72k docs en RAM
- Encodage par batch (256) → conforme aux bonnes pratiques
- Normalisation L2
- Création FAISS si première batch
- Ajout progressif dans l’index
- Export :
  - `faiss.index`
  - `faiss_mapping.json` (liste des UIDs dans l’ordre des vecteurs)

## ✔ Pourquoi pas un index HNSW ?

- HNSW est utile quand tu dépasses 1–2 millions de vecteurs.
- Dans ton cas : ~72 216 documents → FlatIP est optimal.

------

# 2️⃣ **DenseRetriever (dense_retriever.py)**

📄 Fichier utilisé : **dense_retriever.py** 

## ✔ Rôle

Interroger FAISS pour obtenir les documents les plus proches d’une requête utilisateur.

## ✔ Pipeline interne

1. Encoder la requête avec `model.encode_queries`
2. Normaliser → `IndexFlatIP = cosine similarity`
3. Recherche top-k
4. Retour d’un dictionnaire :
   - uid
   - doc_id
   - rank
   - dense_rank_score

## ✔ Justifications (livres)

### *AI Engineering*

Huyen recommande de séparer :

- **retrieval** “cheap but large recall”
- **reranking** “expensive but precise”

Cela améliore les performances globales.

### *RAG-Driven Generative AI*

Rothman montre (chap. 1) que la pertinence sémantique dépend beaucoup du modèle d’embedding → BGE-M3 fait partie des modèles SOTA multilingues.

------

# 3️⃣ **Reranker Cross-Encoder (bge_reranker.py)**

📄 Fichier utilisé : **bge_reranker.py** 

## ✔ Rôle

Réordonner les résultats du retriever pour améliorer la précision.

Le Cross-Encoder lit **(requête, document)** simultanément et attribue un score précis.

## ✔ Pourquoi un Reranker ?

Huyen (Chap. 6, “Retrieval Optimization”) insiste :

- Les retrievers denses = **high recall**, low precision
- Les cross-encoders = **high precision**, expensive

La combinaison = RAG professionnel.

Rothman (Chap. 5) l’appelle **Hybrid Adaptive RAG**.

## ✔ Sortie

Le reranker :

- trie les documents
- ajoute `rerank_score` dans metadata
- retourne des `Document` LangChain → parfait pour l’Étape 4

------

# 4️⃣ **Construction du Contexte (context_builder.py)**

📄 Fichier utilisé : **context_builder.py** 

## ✔ Rôle

Assembler un contexte propre, structuré et compact pour le LLM, en utilisant désormais le format TOON, spécialement conçu pour les entrées structurées des modèles de langage.

Ce format remplace l’ancien contexte Markdown afin d'améliorer la précision, réduire la consommation de tokens et renforcer la robustesse du pipeline RAG.

## ✔ Capacités principales

### 🔹 **1. Nettoyage standardisé du texte**

Chaque document est :

* Nettoyé (suppression d'espaces, sauts de lignes parasites…)
* Normalisé
* Dédupliqué pour éviter les répétitions inutiles

---

### 🔹 **2. Agrégation multi-documents**

Les documents pertinents issus du retriever dense + reranker sont assemblés en une structure tabulaire TOON.

Exemple de structure utilisée :

```toon
documents[N]{uid,dense_score,rerank_score,content}:
  1234,0.9123,0.8777,"Contenu nettoyé…"
  5678,0.8321,0.8012,"Contenu nettoyé…"
```

Chaque **ligne représente un événement unique** avec ses métadonnées essentielles.

---

### 🔹 **3. Encodage TOON via la librairie officielle (`toon.encode`)**

Le context builder construit un dictionnaire Python contenant :

* UID
* Scores (dense + rerank)
* Contenu nettoyé

Puis il génère la sortie TOON via :

```python
from toon import encode
toon_context = encode(data)
```

Cela garantit :

* une structure TOON correcte et stable,
* un format lisible par le LLM,
* un contexte beaucoup plus compact que le Markdown initial.

---

### 🔹 **4. Gestion intelligente des tokens**

Le contexte final TOON est :

* mesuré avec `tiktoken`,
* tronqué proprement si le budget (2000 tokens par défaut) est dépassé,
* renvoyé pour intégration dans le prompt final.

Ce trim est effectué **après encodage**, pour préserver la cohérence TOON.

## ✔ Pourquoi ces choix ?

🔹 *AI Engineering* (Chap. 6, p. 253) :

> “La qualité du contexte est plus importante que la quantité.”

🔹 Rothman recommande le *context packing* + *token budgeting*.

Le builder applique exactement cette philosophie.

------

# 🧩 **Résumé du Pipeline Étape 3**

| Étape | Composant            | But                                    | Source             |
| ----- | -------------------- | -------------------------------------- | ------------------ |
| 1     | build_dense_index.py | Encoder + indexer les documents        | BGE-M3             |
| 2     | dense_retriever.py   | Top-k dense retrieval                  | FAISS              |
| 3     | bge_reranker.py      | Réordonner avec cross-encoder          | BGE reranker       |
| 4     | context_builder.py   | Construire un contexte propre optimisé | LangChain Document |

