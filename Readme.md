# 📚 Puls-Events — Advanced RAG System for Cultural Event Recommendation

Projet 11 — Master Data Engineer (OpenClassrooms)

---

## 🎯 Objectif du projet

Ce projet consiste à concevoir, implémenter et évaluer un **système RAG (Retrieval-Augmented Generation) avancé** capable de répondre à des questions utilisateur et de recommander des événements culturels à partir d’un corpus OpenAgenda.

Le projet met l’accent sur :

- une **architecture RAG modulaire et robuste**,
- la **réduction des hallucinations** via un contexte strictement contrôlé,
- une **évaluation quantitative complète** de chaque composant (retriever, reranker, contexte, LLM),
- une **approche orientée production**, testée et instrumentée.

---

## 🧠 Vue d’ensemble de l’architecture

Le pipeline RAG suit les étapes suivantes :

1. Préparation et nettoyage des données OpenAgenda  
2. Indexation vectorielle FAISS (embeddings denses)  
3. Retrieval dense + reranking cross-encoder  
4. Construction d’un contexte optimisé (format TOON)  
5. Génération de réponse via LLM (Gemini)  
6. Évaluation technique et end-to-end  

Chaque étape est **documentée séparément** dans un README dédié.

---

## 🗂️ Organisation de la documentation

La documentation du projet est volontairement **découpée par étapes**, afin de garantir :

- lisibilité,
- traçabilité,
- conformité aux attentes OpenClassrooms.

### 🔹 Étapes de conception et d’implémentation

| Étape       | Contenu                                               | Documentation                                  |
| ----------- | ----------------------------------------------------- | ---------------------------------------------- |
| **Étape 1** | Définition du besoin, corpus et stratégie RAG         | [`Readme Etape 1.md`](.docs/Readme%20Etape%201.md) |
| **Étape 2** | Préprocessing des données OpenAgenda                  | [`Readme Etape 2.md`](.docs/Readme%20Etape%202.md) |
| **Étape 3** | Index vectoriel, retriever, reranker, context builder | [`Readme Etape 3.md`](.docs/Readme%20Etape%203.md) |
| **Étape 4** | Pipeline RAG complet avec LangChain                   | [`Readme Etape 4.md`](.docs/Readme%20Etape%204.md) |

---

### 🔹 Évaluation du système RAG

Chaque composant critique est évalué **indépendamment**, puis **en bout en bout**.

| Composant évalué | Documentation                                                |
| ---------------- | ------------------------------------------------------------ |
| Retriever        | [`Readme Evaluation - Retriever.md`](.docs/Readme%20Evaluation%20-%20Retriever.md) |
| Reranker         | [`Readme Evaluation - Reranker.md`](.docs/Readme%20Evaluation%20-%20Reranker.md) |
| Context Builder  | [`Readme Evaluation - Context Builder.md`](.docs/Readme%20Evaluation%20-%20Context%20Builder.md) |
| LLM-as-a-Judge   | [`Readme Evaluation - LLM-as-Judge.md`](.docs/Readme%20Evaluation%20-%20LLM-as-Judge.md) |

Ces documents détaillent :
- les métriques utilisées,
- les protocoles d’évaluation,
- l’interprétation des résultats,
- les limites identifiées.

---

## ▶️ Exécution rapide (aperçu)

> Les instructions complètes sont détaillées dans les READMEs des étapes concernées.

```bash
# Installation
pip install -r requirements.txt

# Lancer l’interface utilisateur
chainlit run app.py
````

---

## 🧪 Tests & Qualité

Le projet inclut :

* des tests unitaires ciblant les invariants RAG,
* des tests d’intégration sur composants réels (FAISS, reranker, context builder),
* une instrumentation complète via Langfuse / LangSmith.

Les détails sont documentés dans les READMEs d’évaluation.

---

## 🚧 Limites et perspectives

Les limites connues et pistes d’amélioration (RAG agentique, guardrails supplémentaires, déploiement cloud) sont discutées dans les documents d’évaluation et de conclusion.

---

## 👤 Auteur

Projet réalisé par **Rudy Desplan**
Dans le cadre du **Master Data Engineer — OpenClassrooms**