"""
Dans ce fichier python, quelques exemples de code très simple pour le retrevial
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

#Exemple de la partie Retrevial => 1:doc pertinent / 0: Doc non pertinent

y_pred = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
y_true = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]


metrics = np.array([precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)])
name = np.array(["Precision", "Recall", "f1_score"])

data_metrics = pd.DataFrame({"name": name, "score": metrics})

print("Précision :", precision_score(y_true, y_pred))
print("Rappel    :", recall_score(y_true, y_pred))
print("F1-score  :", f1_score(y_true, y_pred))

#on peut ensuite afficher nos résulats sur un graphique

sns.barplot(data=data_metrics, x='name', y='score', palette="pastel")
plt.title("Métrique de base pour le Retrevial")
plt.show()

#################################################################################################################

##on prend maintenant compte des métriques avec l'ordre 

#Impossible de le tester, pas de compilateur sur ce pc et besoin de droit d'admin
"""
from ir_measures import nDCG, MRR, MAP
from ir_measures import Qrel, ScoredDoc, calc_aggregate

# === 1. Vérité terrain : pertinence réelle des documents pour la requête Q1 ===
qrels = [
    Qrel('Q1', 'doc1', 3),   # très pertinent
    Qrel('Q1', 'doc2', 2),   # pertinent
    Qrel('Q1', 'doc3', 0),   # pas pertinent
    Qrel('Q1', 'doc4', 1),   # moyennement pertinent
]

# === 2. Résultat du modèle : prédiction du classement des documents ===
retrieved = [
    ScoredDoc('Q1', 'doc2', 0.95),  # prédit très pertinent
    ScoredDoc('Q1', 'doc3', 0.80),  # erreur : non pertinent mis en haut
    ScoredDoc('Q1', 'doc1', 0.75),  # vrai très pertinent, mais pas en 1ère position
    ScoredDoc('Q1', 'doc4', 0.60),  # un peu pertinent
]

# === 3. Calcul des métriques ===
results = calc_aggregate([nDCG@3, MRR@3, MAP@3], qrels, retrieved)

# === 4. Affichage ===
print("Résultats d'évaluation du modèle (top 3 docs) :")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

"""