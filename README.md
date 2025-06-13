# Évaluation d’un système RAG

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/0005cb44-e213-42ff-9dae-312b49c0b191_2000x1190.jpeg" alt="Image représentant un RAG" width="400"/>

## Sommaire

- [Introduction](#introduction)
- [Objectifs](#objectifs)
- [Solutions mises en place](#solutions-mises-en-place)
  - [1. Création d’un dataset d’évaluation](#1--la-création-dun-dataset-dévaluation)
  - [2. Embeddings](#2--lembeddings)
  - [3. Chunking](#3--chunking)
  - [4. Retrieval](#4--retrieval)
  - [5. Re-ranking](#5--re-ranking)
  - [6. Réponses du LLM](#6--les-réponses-du-llm)
- [Applications Python](#les-applications-en-python)
  - [Retrieval](#python-retrieval) 
  - [Sans LLM](#python-retrieval-sans-llm)
  - [Avec RAGAS](#ragas-evaluation)


## Introduction

Ce dépôt regroupe différentes solutions pour implémenter et évaluer un système RAG à l’aide de plusieurs outils, dans le but de comparer leur performance et leur pertinence.

**Documents supplémentaires créés**

- [Carte mentale (à finir)](https://www.canva.com/design/DAGpTs9hinY/zjeUoHolQ4k2BgugcSM48Q/view?utm_content=DAGpTs9hinY&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hecde141282)

## Objectifs

Ce projet a pour objectif d’évaluer en profondeur les différentes composantes d’un système RAG (Retrieval-Augmented Generation), en testant plusieurs méthodes à chaque étape de la chaîne de traitement.

L’évaluation porte sur :

- **La création d’un dataset d’évaluation** adapté aux besoins du projet.
- **La qualité des embeddings**, en comparant différentes méthodes de vectorisation des documents.
- **Les stratégies de chunking**, afin d’analyser leur impact sur la pertinence des passages extraits.
- **Les techniques de retrieval**, en testant divers modèles de recherche d’information.
- **Les méthodes de re-ranking**, pour optimiser l’ordre des documents récupérés.
- **L’évaluation finale des réponses générées**, à l’aide de métriques fournies par des outils comme RAGAS (factualité, complétude, etc.).

L’objectif global est de comparer ces différentes approches pour identifier les combinaisons les plus efficaces et pertinentes dans un cadre d’utilisation concret.

## Solutions mises en place

### 1- La création d’un dataset d’évaluation

blaaaaa

### 2- L'Embeddings

blaaaaa

### 3- Chunking

blaaaaa

### 4- Retrieval

```metrics_simple_retrieval.py```
- Regroupement des différentes méthodes dans la carte mentale

Le principal problème est de savoir si un document **est pertinent ou ne l'est pas** pour nos évaluations. Étiqueter manuellement la pertinence des documents est trop coûteux. Pour pallier cela, on utilise un LLM juge, capable d’annoter chaque document avec un score de pertinence :

| Score | Signification         |
| ----- | --------------------- |
| 3     | Très pertinent        |
| 2     | Pertinent             |
| 1     | Moyennement pertinent |
| 0     | Pas pertinent         |

  [Exemple en python](#python-retrieval)

### 5- Re-ranking

```metrics_simple_retrieval.py``` (dans la deuxième partie)
- [Exemple en python](#python-retrieval)

### 6- Les réponses du LLM

Deux méthodes possibles : 

- **Métrique sans utilisation de LLM**
- **Métrique avec utilisation de LLM**

Pour les métriques qui n'utilisent pas de **LLM**, on retrouve la liste suivante (expliqué dans la carte mentale) : 
- Bleu score
- Rouge score
- BERTScore
- Meteor
- MoverScore

[Implémentation en Python](#python-evaluation-reponse-sans-LLM)

Pour les métriques qui utilisent un **LLM**, on retrouve la liste suivante (expliqué dans la carte mentale) : 
- RAGAS -> de nombreuses métriques qui analysent les réponses sur plusieurs points ([plus d'info](#ragas-evaluation))


## Les applications en Python

### Python Retrieval 

Le fichier python correspondant : ```metrics_for_LLM_response_reference.py```

On discerne deux types de métriques : 
- Celles qui prennent compte du **classemment**
- Celles qui ne prennent pas compte du **classemment**

Pour celle qui ne prennent pas en compte le classement (début du programme) :
- Précision
- Rappel
- F1-score

**PS** : Utilisation du module **sklearn.metrics**

Données : 

``` python
y_pred = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
y_true = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
```
- 1: Doc pertinent
- 0: Doc non pertinent

Pour celle qui prennent en compte le classement (fin du programme) :
- MRR (Mean Reciprocal Rank)
- Gain cumulatif actualisé normalisé

Ces métriques évaluent si les documents les plus pertinents sont bien en haut du classement.

Utilisation de ```ir_measures``` : ⚠️ nécessite un compilateur pour lancement

### Python évaluation réponse sans LLM

Le fichier python correspondant : ```metrics_simple_retrieval.py```

Dans ce fichier, à partir d'une réponse de référence et d'une question génerée, on calcule les différentes métriques que l'on affiche dans un graphique à barres.

Analyse des résultats : 
- Bleu score ne comprends pas les synonyme (à part si on veut des mots clés, pas très utile)
- Rouge score intéressant : 🛑 Répétition de mot sans sens augmente le score ?
- BERTScore : rien à re-dire

### RAGAS evaluation

Le fichier python correspondant :
- ```evaluate_llm_with_ragas_local.py``` (utilisation d'un LLM local)
- ```evaluate_llm_with_ragas_aws.py``` (utilisation d'un LLM sur AWS)

Pour modifier les données de traitement : ```./data/input```

Dans ce fichier, à partir d'une réponse de référence et d'une question génerée, on calcule les différentes métriques que l'on affiche dans un graphique à bar.

🛑 Problème de timeOut, le LLM prends trop de temps pour répondre, on est donc obligé de l'empecher de travailler sur plusieurs métriques simultanément 

``` Python 
my_config = RunConfig(
    max_workers=1,
)
```

**PS** : je n'ai trouvé que cette solution 

Les résultats des scores sont renseignés dans un fichier exel :  ```./data/ouput```

Il y a dans ce fichier l'évaluation des métriques suivantes : 
- answer_correctness
- answer_relevancy
- faithfulness
- context_precision
- context_recall


🛑 Le temps de réponse sur l'Evaluation est **très long**

🛑 Analyse des résultats à faire 

## A faire ... 


