# # Étape 1 — Définition du besoin, du corpus et de l’architecture RAG

## 🎯 1. Objectif du projet

Ce projet consiste à concevoir et déployer un **chatbot intelligent** capable de :

- répondre aux questions des utilisateurs concernant des événements (culturels, locaux, en ligne) ;
- fournir des recommandations personnalisées à partir d’un corpus structuré ;
- générer des réponses enrichies à partir des données stockées dans une base vectorielle (**FAISS-CPU**) ;
- combiner **recherche sémantique** et **génération** via les modèles **Gemini 2.5**.

### Pourquoi un système RAG ?

Comme le décrivent Rothman (chap. *Naïve RAG → Advanced RAG*) et Huyen (chap. 6, *RAG & Retrieval Optimization*) :

- Les LLM seuls **hallucinent** → le RAG injecte des sources fiables.
- Les LLM ne connaissent pas les données locales → le RAG donne accès à un corpus privé.
- Les réponses doivent refléter des **informations dynamiques** (dates, lieux, statuts d’événements).
- Le RAG permet une **personnalisation**, ce qui est idéal pour la recommandation.

Ainsi, le RAG est **le paradigme recommandé** pour un chatbot basé sur une base de connaissances dynamique et structurée.

------

# ## 2. Description du corpus documentaire

Le corpus utilisé dans ce projet est un **fichier Parquet** contenant **56 champs** décrivant des événements culturels, professionnels ou communautaires.
 Chaque enregistrement correspond à **un événement unique**, accompagné de métadonnées textuelles, géographiques, temporelles, organisationnelles et médiatiques.

Ce corpus est adapté à un système RAG car il offre :

- un **niveau de granularité élevé**, idéal pour transformer chaque événement en un document vectorisable ;
- une combinaison de données **textuelles, structurées et géographiques** ;
- des champs permettant la **recommandation personnalisée** (âge, localisation, catégorie, mots-clés, accessibilité) ;
- des attributs permettant d’optimiser la **recherche sémantique + filtrage**.

------

## ### 📌 Liste complète des 56 champs du corpus

### **1. Identité de l’événement**

1. **uid** — Identifiant unique
2. **slug** — Slug
3. **canonicalurl** — URL canonique
4. **title_fr** — Titre
5. **description_fr** — Description courte
6. **longdescription_fr** — Description longue
7. **conditions_fr** — Détail des conditions
8. **keywords_fr** — Mots clés
9. **category** — Catégorie
10. **status** — État de l'événement
11. **registration** — Informations d’inscription
12. **links** — Liens additionnels

------

### **2. Informations temporelles**

1. **updatedat** — Dernière mise à jour
2. **daterange_fr** — Résumé des horaires
3. **firstdate_begin** — Première date (début)
4. **firstdate_end** — Première date (fin)
5. **lastdate_begin** — Dernière date (début)
6. **lastdate_end** — Dernière date (fin)
7. **timings** — Horaires détaillés

------

### **3. Accessibilité**

1. **accessibility** — Code d'accessibilité
2. **accessibility_label_fr** — Libellé accessibilité

------

### **4. Informations géographiques**

1. **location_uid** — Identifiant du lieu
2. **location_coordinates** — Coordonnées géographiques (point géo)
3. **location_name** — Nom du lieu
4. **location_address** — Adresse
5. **location_district** — Arrondissement
6. **location_insee** — Code INSEE
7. **location_postalcode** — Code postal
8. **location_city** — Ville
9. **location_department** — Département
10. **location_region** — Région
11. **location_countrycode** — Pays (code)
12. **location_image** — Image du lieu
13. **location_imagecredits** — Crédits de l’image du lieu
14. **location_phone** — Téléphone du lieu
15. **location_website** — Site web du lieu
16. **location_links** — Liens associés au lieu
17. **location_tags** — Tags du lieu
18. **location_description_fr** — Description du lieu
19. **location_access_fr** — Accès / Itinéraire

------

### **5. Médias associés**

1. **image** — Image principale
2. **imagecredits** — Crédits de l’image
3. **thumbnail** — Miniature
4. **originalimage** — Image source

------

### **6. Type d’événement et participation**

1. **attendancemode** — Événement physique ou en ligne
2. **onlineaccesslink** — Lien d'accès en ligne

------

### **7. Conditions d’âge**

1. **age_min** — Âge minimum
2. **age_max** — Âge maximum

------

### **8. Informations sur l’agenda d’origine**

1. **originagenda_title** — Titre de l’agenda d’origine
2. **originagenda_uid** — UID de l’agenda d’origine

------

### **9. Informations sur le contributeur**

1. **contributor_email** — Email du contributeur
2. **contributor_contactnumber** — Téléphone du contributeur
3. **contributor_contactname** — Nom du contributeur
4. **contributor_contactposition** — Fonction du contributeur
5. **contributor_organization** — Organisation contributrice

------

### **10. Informations géopolitiques**

1. **country_fr** — Nom du pays

------

## ### 🎯 Pourquoi ce corpus est optimal pour un RAG ?

### ✔ *RAG-Driven Generative AI* (Rothman) souligne que les meilleurs corpus RAG :

- combinent **texte riche + métadonnées** (conditions remplies ici) ;
- permettent une **indexation modulable** (événement = chunk) ;
- facilitent une **recherche hybride** (sémantique + filtres géographiques et temporels).

### ✔ *AI Engineering* (Huyen) détaille que :

- la qualité du **contexte injecté dans le LLM** détermine la qualité des réponses ;
- les données structurées sont idéales pour des systèmes RAG “augmentés” avec filtres ;
- la diversité des champs permet d’améliorer la **personnalisation des recommandations**.

------

## ### 🎯 Rôle de ces données dans le futur pipeline RAG

Ce corpus permettra :

- la **recherche vectorielle** sur les champs textuels enrichis (titres, descriptions, lieux, accessibilité…) ;
- la **recommandation personnalisée** via les métadonnées :
  - âge, ville, distance géographique, catégorie, dates ;
- des **prompts contextualisés** incluant horaires, lieux et informations pratiques ;
- la génération de **réponses fiables** avec justification (source : UID, titre).

------

# ## 3. Type d’application RAG développée

L’application combine deux modes :

### 🧠 **Mode 1 — Chatbot de réponses augmentées**

- Les documents pertinents sont récupérés via FAISS.
- Le LLM Gemini génère une réponse structurée, factuelle et contextualisée.

### 🎯 **Mode 2 — Recommandation personnalisée d’événements**

Basée sur :

- similarité sémantique (embeddings)
- préférences utilisateur (âge, ville, thèmes)
- métadonnées (dates, accessibilité, type d’événement)

Ce modèle correspond à un **Hybrid RAG**, comme recommandé par Huyen pour les assistants recommandateurs.

------

# ## 4. Stratégie RAG retenue

Nous adoptons un **RAG modulaire** :

### 4.1. Chunking

Chaque événement génère **un document unique**, enrichi avec :

- description courte
- description longue
- informations pratiques
- coordonnées
- accessibilité
- conditions et restrictions

### 4.2. Encodage des embeddings

📌 **Modèle utilisé : `BAAI/bge-m3`**
 → Modèle d’embeddings dense multilingue, optimisé pour la recherche sémantique et les textes longs, utilisé pour la construction de l’index FAISS dans ce projet.

Justification :

- Qualité de représentation sémantique reconnue dans les benchmarks de type MTEB/MIRACL ;

- Support natif des contextes longs (jusqu’à 8192 tokens), adapté à des documents événementiels riches ;

- Compatibilité avec FAISS CPU, garantissant une exécution locale reproductible sans dépendance GPU ;

- Capacité à fournir un fort rappel (high recall), indispensable pour une architecture RAG combinant retrieval dense et reranking cross-encoder.

### 4.3. Vector Store : FAISS-CPU

Exigence du projet → **FAISS CPU** sera utilisé.

Justification technique  :

- rapide en local,
- adapté aux prototypes comme aux déploiements légers,
- excellent pour top-k search et inner-product similarity.

### 4.4. Retrieval amélioré 

- **top-k retrieval** (k = 5–10)
- **reranking sémantique dédié** (cross-encoder, analysé et contrôlé)
- **hybrid filtering** (ville, âge, dates)
- **diffusion de contexte** dans le prompt final

------

# ## 5. Choix du modèle générateur Gemini

Les modèles disponibles dans ce projet :

| Modèle                    | Type     | TPM  | RPM      | RPD      | Usage recommandé                |
| ------------------------- | -------- | ---- | -------- | -------- | ------------------------------- |
| **gemini-2.5-flash-live** | API Live | 1M   | illimité | illimité | Chatbot interactif en streaming |
| **gemini-2.5-flash**      | Texte    | 250K | 10       | 250      | Réponses longues, coût modéré   |

- **RPM (Requests Per Minute)** = nombre de requêtes que tu peux envoyer **par minute** à ce modèle.
- **TPM (Tokens Per Minute)** = nombre total de tokens (entrée + sortie) que tu peux consommer **par minute** avec ce modèle.
- **RPD (Requests Per Day)** = nombre de requêtes que tu peux envoyer **par jour** pour ce modèle.

### Décision :

💡 **Le projet GitHub exposera les 3 options**, afin que l’utilisateur choisisse selon ses besoins :

- **Lite** → vitesse & coût minimal
- **Flash** → meilleur compromis
- **Flash-Live** → interactions streaming type chat

------

# ## 6. Architecture générale IO (Input → Retrieval → Generation → Output)

### 🔵 **INPUT**

- question utilisateur
- préférences (ville, âge, catégorie)
- paramètres : k, modèle Gemini, filtres

### 🟠 **PROCESS**

1. embedding de la requête (gemini-embedding-001)
2. recherche vectorielle FAISS CPU
3. application des filtres (dates, ville, âge)
4. reranking
5. construction du prompt RAG
6. génération via Gemini (modèle choisi)

### 🟢 **OUTPUT**

- réponse contextualisée
- liste d’événements pertinents
- liens, infos pratiques, horaires
- sources (uid événement, titre)

Ce pipeline suit la structure décrite par Rothman (RAG Pipeline en 3 modules)

 et par Huyen (Context Construction → Model → Output).

