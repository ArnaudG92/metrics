# Évaluation d’un système RAG

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/0005cb44-e213-42ff-9dae-312b49c0b191_2000x1190.jpeg" alt="Image représentant un RAG" width="400"/>

## Sommaire

- [Introduction](#introduction)
- [Objectifs](#objectifs)
- [Solutions mises en place](#solutions-mises-en-place)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Auteur](#auteur)

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


## A faire ... 


