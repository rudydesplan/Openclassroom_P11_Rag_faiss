# 📘 **README — Étape 4 : Conception du Pipeline RAG Complet avec LangChain**

## 🎯 Objectif de l’Étape 4

L’objectif de cette étape est de transformer les composants développés en Étapes 1–3 en un **système RAG complet**, orchestré par LangChain, capable de :

1. Recevoir une requête utilisateur,
2. Extraire les documents les plus pertinents via un **retriever dense**,
3. Affiner la pertinence avec un **reranker cross-encoder**,
4. Construire un **contexte optimisé(TOON)**,
5. L’envoyer à un **modèle de génération** (LLM),
6. Retourner une réponse structurée, fiable et traçable.
7. Maintenir une conversation naturelle avec l’utilisateur.

------

# 🧱 **1. Rappel de l’architecture construite en Étapes 1–3**

On disposes maintenant :

### ✔ d’un dataset nettoyé (Étape 2)

### ✔ d’un index vectoriel FAISS construit à partir de BGE-M3 (Étape 3)

→ via `build_dense_index.py`

### ✔ d’un retriever dense local

→ `dense_retriever.py`

### ✔ d’un reranker cross-encoder BGE

→ `bge_reranker.py`

### ✔ d’un context builder adaptable et token-aware

→ `context_builder.py`

Le context builder génère maintenant un **contexte au format TOON**, via la librairie officielle `toon.encode()`.
TOON est un format tabulaire compact conçu pour les LLM, permettant :

* d’envoyer plus de données dans un budget de tokens réduit,
* d’améliorer la précision des modèles,
* de réduire les hallucinations.

Ces trois composants sont **modulaires, testables et indépendants de LangChain**.




📌 **Étape 4 consiste uniquement à orchestrer proprement ces modules avec LangChain.**

------

# 🔥 **2. Pourquoi introduire LangChain maintenant ?**

### 1. **Séparation des responsabilités**

Les composants fondamentaux sont déjà stables :

- indexation,
- retrieval,
- reranking,
- génération de contexte.

Il est donc logique de passer à un orchestrateur haut niveau.

### 2. **LCEL : meilleure traçabilité et meilleure observabilité**

LangChain Expression Language permet de :

- composer des pipelines lisibles (`|`),
- logguer chaque étape,
- ajouter facilement des étapes d'évaluation ou de caching,
- préparer la mise en production.

### 3. **Interopérabilité entre LLMs**

LangChain simplifie :

- les prompts structurés,
- le changement de modèle (OpenAI, HF, Gemini, Mistral…),
- l’instrumentation automatique via callbacks.

📌 **Étape 4 est le bon moment car tu as enfin tous les blocs nécessaires.**

------

# 🏗️ **3. Architecture RAG finale (Étape 4)**

Voici la pipeline exigée à l’Étape 4 :

```
             ┌──────────────────────┐
             │    User Query        │
             └───────────┬──────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│    1. LangChain Dense Retriever (wrapper)               │
│        → utilise dense_retriever.py                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│    2. LangChain Reranker (wrapper)                      │
│        → utilise bge_reranker.py                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│    3. LangChain Context Builder (wrapper)               │
│        → utilise context_builder.py                     │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│    4. PromptTemplate                                    │
│        → structure la requête finale                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│    5. LLM                                               │
│        → Gemini                                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Memory : RunnableWithMessageHistory                  │
│  → wrap la chaîne RAG complète                          │
└─────────────────────────────────────────────────────────┘	
```

------

# 🧩 **4. Ce qu'on doit implémenter en Étape 4**

Voici précisément les éléments à ajouter :

------

## ✔ 4.1. **Wrapper LangChain du Dense Retriever**

Créer : `langchain_dense_retriever.py`

Il doit :

- hériter de `BaseRetriever`
- appeler ton `DenseRetriever.search()`
- charger le texte original via l’UID
- renvoyer une liste de `Document` LangChain

Pourquoi ?
 Car LangChain exige que tout composant de récupération implémente cette interface pour être compatible LCEL.

------

## ✔ 4.2. **Wrapper LangChain du Reranker**

Créer : `langchain_reranker.py`

- transforme `List[Document]` → `List[Document]`
- utilise ton `BGEReranker.rerank()`
- injecte `rerank_score` dans metadata

Ce module devient un **RunnableLambda** dans la pipeline LCEL.

------

## ✔ 4.3. **Wrapper LangChain du Context Builder**

Créer : `langchain_context_builder.py`

- prend les Documents
- appelle `build_context(..., output_format="toon")`
- renvoie un string
- devient un `RunnableLambda`

------

## ✔ 4.4. **Le pipeline final LCEL : rag_pipeline.py**

Il assemble :

```python
chain = (
    retriever
    | reranker
    | context_builder
    | prompt
    | llm
    | output_parser
)
```

Ce pipeline :

- est traçable,
- composable,
- testable,
- modulaire.

------

# 🗨️ **5. Interaction utilisateur : ajout de la mémoire conversationnelle**

C’est la **nouvelle mise à jour majeure**.

OpenClassrooms exige :

> “Un chatbot capable de fournir des recommandations **ET d’interagir avec l’utilisateur**.”

Cela nécessite une **mémoire d’historique**, afin que le bot comprenne :

* les questions de suivi (“Et pour les enfants ?”)
* les ellipses (“Et demain ?”)
* les corrections (“Finalement je préfère Lyon”)

Nous utilisons donc :

### ✔ `ConversationBufferMemory`

pour stocker l’historique complet des échanges.

### ✔ `RunnableWithMessageHistory`

pour intégrer la mémoire dans LCEL.

### Exemple d’intégration :

```python
conversation_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: memory,
    input_messages_key="query",
    history_messages_key="chat_history",
)
```

Et lors de l’appel :

```python
response = rag.invoke(
    {"query": "Et pour les enfants ?"},
    config={"configurable": {"session_id": "user42"}}
)
```

➡️ Le bot comprend la continuation de la conversation.

------

# 📑 **6. Prompt Engineering Étape 4**

Le prompt doit inclure :

### 1. Le contexte (rempli dynamiquement via LCEL)

### 2. Des instructions claires

### 3. Une demande de transparence (option OC)

### 4. Des règles de refus si le contexte ne suffit pas

### 5. Un format de sortie structuré

#### 🔶 Intégration du format TOON dans le prompt

Le contexte injecté dans le prompt est désormais encodé au format TOON.
Le prompt inclut un code fence dédié :

````markdown
```toon
{context}
````

------

# 🧪 **7. Tests et évaluation**

Pour l’Étape 4, OC demande :

- des tests unitaires sur les modules critiques,
- des tests fonctionnels end-to-end du pipeline RAG,
- (préparation) Évaluation via RAGAS à l’Étape 5.

LangChain facilite cela car chaque étape est indépendante.

------

# 🧠 **8. Justification des choix techniques**

Voici les justifications basées sur les livres fournis.

------

### ✔ Pourquoi un **retriever dense BGE-M3** ?

- Modèle multilingue, adapté à des textes longs (8192 tokens)
- Embeddings plus robustes que ceux des modèles basés uniquement sur SBERT
- SOTA dans plusieurs benchmarks MIRACL/MTEB

------

### ✔ Pourquoi un **reranker cross-encoder** ?

> “Dense retrieval = rappel élevé, reranker = précision élevée.
>  La combinaison est indispensable pour des RAG de qualité.”

------

### ✔ Pourquoi un **context builder personnalisé** ?

LangChain fournit un "stuff" et "map_reduce", mais :

- ils ne nettoient pas le texte
- ils ne gèrent pas la déduplication
- ils ne sont pas optimisés pour ton dataset

Ton choix est donc **supérieur à la solution générique**.

------

### ✔ Pourquoi utiliser le format TOON pour le contexte **?

TOON est un format conçu pour les LLMs : il est tabulaire, compact et explicite.  
Il permet :

- d’intégrer **plus de contexte dans le même budget token**,  
- d’améliorer la capacité du LLM à comparer les documents,  
- de réduire les hallucinations grâce à une structure stricte,  
- de faciliter l’analyse des scores (`dense_score` / `rerank_score`).  

L’intégration de TOON à l’Étape 4 améliore significativement les performances globales du pipeline RAG.

------

### ✔ Pourquoi utiliser LCEL pour le pipeline ?

Parce que LCEL :

- remplace les anciens chains complexes,
- permet une composition propre (`|`),
- ajoute instrumentation, caching, observabilité,

------

# 🎉 **Conclusion — Étape 4 terminée**

On as maintenant :

- une architecture RAG complète,
- des composants modulaires et robustes,
- un pipeline LCEL conforme aux standards de 2025,
- une séparation parfaite backend/orchestration,
- une base solide pour l’Étape 5 (évaluation, UI, déploiement).