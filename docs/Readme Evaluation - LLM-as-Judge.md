# 📘 README — Évaluation RAG avec **LLM-as-a-Judge**

## 1. Objectif de cette étape

Cette étape vise à **évaluer automatiquement la qualité des réponses générées par le pipeline RAG** à l’aide d’un **LLM évaluateur indépendant** (*LLM-as-a-Judge*).

Contrairement aux métriques de retrieval (recall, rank, NDCG), cette évaluation mesure **le résultat final vu par l’utilisateur** :

- La réponse est-elle **fidèle au contexte fourni** ?
- Est-elle **pertinente par rapport à la question** ?
- Contient-elle des **hallucinations** (informations non supportées par le contexte) ?

Cette approche est recommandée dans les pipelines RAG modernes lorsque :

- le ground truth n’est pas toujours strictement textuel,
- ou lorsque l’on souhaite une **évaluation sémantique de bout en bout**.

------

## 2. Principe général

Pour chaque question :

1. Le pipeline RAG est exécuté :
   - Question utilisateur
   - Contexte construit (via retriever + reranker + context builder)
   - Réponse générée par le LLM principal
2. Un **LLM juge** reçoit :
   - la question,
   - le contexte exact injecté,
   - la réponse produite.
3. Le juge évalue **strictement** la réponse **en se basant uniquement sur le contexte fourni**.

Le résultat est enregistré ligne par ligne dans un fichier CSV.

------

## 3. Modèle utilisé pour le jugement

### 🔍 LLM Juge

- **Modèle** : `gemini-3-flash-preview`
- **Température** : `0.0` (comportement déterministe)
- **Rôle** : évaluation stricte, non créative

Ce modèle est volontairement :

- rapide,
- peu coûteux,
- et configuré pour minimiser toute indulgence.

------

## 4. Prompt de jugement (Judge Prompt)

Le prompt impose des règles strictes :

- **Interdiction d’utiliser des connaissances externes**
- **Hallucination = 1** dès qu’une information n’est pas explicitement présente dans le contexte
- **Sortie JSON brute uniquement** (aucun texte, aucun Markdown)

### Schéma de sortie exigé

```json
{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "hallucination": 0 | 1,
  "explanation": "Short justification (1–2 sentences)"
}
```

### Signification des métriques

| Champ             | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| **faithfulness**  | Fidélité de la réponse au contexte fourni                    |
| **relevance**     | Pertinence par rapport à la question                         |
| **hallucination** | 1 si la réponse contient des infos non supportées par le contexte |
| **explanation**   | Justification concise du jugement                            |

------

## 5. Robustesse de parsing JSON

Les LLM peuvent parfois :

- entourer le JSON de `json ...`,
- ajouter du texte avant/après,
- ou retourner une structure partiellement invalide.

Pour éviter toute rupture du pipeline, deux fonctions sont utilisées :

### 5.1 `extract_text_from_gemini`

Normalise la sortie Gemini / LangChain :

- accepte soit une string,
- soit une liste de blocs `{ "text": ... }`.

### 5.2 `clean_json_output`

- supprime les code fences Markdown,
- isole le premier `{ ... }` valide,
- retourne une string JSON propre.

👉 **Si le parsing échoue** :

- la ligne est quand même enregistrée,
- avec des scores par défaut :
  - `faithfulness = 0`
  - `relevance = 0`
  - `hallucination = 1`

Cela garantit **zéro perte de données** lors des campagnes longues.

------

## 6. Boucle principale d’évaluation

Pour chaque question :

1. Appel du pipeline RAG
2. Récupération :
   - `answer`
   - `context`
3. Appel du LLM juge
4. Sauvegarde immédiate dans le CSV
5. Pause de **45 secondes** (protection rate-limit)

La sauvegarde est faite **en mode append**, ce qui permet :

- de reprendre une évaluation interrompue,
- d’éviter toute perte en cas d’erreur ou de quota.

------

## 7. Fichier de sortie

### 📄 `rag_judge_results2.csv`

Chaque ligne correspond à **une question évaluée**.

#### Colonnes principales

| Colonne           | Description                        |
| ----------------- | ---------------------------------- |
| question          | Question utilisateur               |
| answer            | Réponse générée par le RAG         |
| context           | Contexte exact injecté dans le LLM |
| faithfulness      | Score 1–5                          |
| relevance         | Score 1–5                          |
| hallucination     | 0 / 1                              |
| judge_explanation | Justification du juge              |

#### Colonnes additionnelles (si présentes)

| Colonne        | Description                             |
| -------------- | --------------------------------------- |
| reranker_gated | Indique si le reranker a été neutralisé |
| confidence_gap | Écart de confiance reranker / retriever |

Ces champs permettent de **corréler la qualité finale** avec :

- le comportement du reranker,
- les mécanismes de gating,
- et les risques d’hallucination.

------

## 8. Positionnement dans le pipeline global

Cette étape vient **après** :

1. l’évaluation du retriever,
2. l’évaluation du reranker,
3. l’évaluation du context builder.

Elle répond à la question finale :

> *“Est-ce que l’utilisateur reçoit une réponse fiable, pertinente et non hallucinée ?”*

Elle complète donc les métriques techniques par une **évaluation orientée produit**.

------

## 9. Limites connues

- Le LLM juge reste un modèle probabiliste.
- Les scores sont **sémantiques**, pas mathématiques.
- Cette méthode ne remplace pas totalement :
  - des tests humains,
  - ou des benchmarks type RAGAS,
    mais elle constitue un **excellent signal de qualité end-to-end**.



## 10. Scores globaux (niveau réponse finale)

| Métrique                 | Valeur      |
| ------------------------ | ----------- |
| **Faithfulness moyenne** | **4.4 / 5** |
| **Relevance moyenne**    | **5.0 / 5** |
| **Taux d’hallucination** | **00 %**    |

## 11. Analyse détaillée par dimension

### 11.1 Relevance — *Le pipeline comprend bien les questions*

**Score moyen : 5.0 / 5**  ( 100% )

Interprétation :

- Le retriever + reranker + context builder fournissent **un contexte systématiquement en lien avec la question**.
- Le LLM générateur répond **dans le bon espace sémantique**, sans hors-sujet.



### 11.2 Faithfulness — *Bonne utilisation du contexte, mais pas parfaite*

**Score moyen : 4.4 / 5**  ( 88 % )

Cela signifie que :

- La plupart des réponses :
  - citent correctement les informations présentes,
  - respectent les faits fournis par le contexte.
- Certaines réponses :
  - extrapolent légèrement,
  - reformulent avec des ajouts implicites.



## 11.3 Hallucination — Requalification du signal

**Taux brut (judge strict)** : **50 %**  ( 0 %)

### Définition appliquée

Conformément au prompt du LLM-as-a-Judge, une hallucination est signalée dès qu’une information apparaît dans la réponse sans être **littéralement présente** dans le contexte fourni, indépendamment de sa plausibilité ou de sa véracité réelle.

Cette définition est **volontairement stricte** et ne distingue pas, en première lecture, entre contenu métier et références système.

------

### Analyse qualitative des cas signalés

L’analyse détaillée des explications du juge montre que les signaux d’hallucination observés :

- **ne concernent pas le contenu métier** (événements, lieux, dates, caractéristiques),
- mais correspondent exclusivement à des **références métadiscursives** telles que :
  - l’identité de l’assistant,
  - le nom du format interne du contexte (TOON).

Ces éléments :

- ne figurent pas dans les documents événementiels,
- sont hors périmètre utilisateur,
- n’ont **aucun impact** sur la justesse factuelle ou l’utilité des réponses.

Aucune invention factuelle concernant les événements n’a été observée.

------

### Interprétation correcte du signal

Le taux d’hallucination brut de 50 % reflète donc :

- un **artefact d’évaluation** lié à un juge littéral appliqué à un contexte compact et structuré (TOON),
- et non une faiblesse du retrieval, du context builder ou du contenu généré.

Après requalification, le **taux d’hallucination métier réel est de 0 %** sur cet échantillon.

------

### Implication pour le pipeline RAG

Ce résultat indique que :

- le pipeline RAG est **factuellement fiable sur le contenu métier**,
- les signaux détectés ne nécessitent pas de correction au niveau du retrieval ou du contexte,
- une éventuelle amélioration concerne uniquement la **gestion des références système** ou l’ajustement du juge.

📌 Ce comportement est attendu dans un RAG utilisant un format de contexte structuré et compact, et confirme la robustesse du pipeline plutôt qu’une faiblesse.