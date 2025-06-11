"""
Métrique pour évaluer la réponse à l'aide d'un réponse de référence (sans appel à un LLM)
"""
# Import des librairies
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
#from moverscore import word_mover_score

# Pré-traitement simple (minuscules, suppression ponctuation, tokenisation)
def preprocess(text):
    import re
    # enlever ponctuation (simplifié)
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return tokens

# -------------------------------
# 1) Calcul BLEU (score sur tokens)
# -------------------------------
# BLEU classique, lissage pour éviter le score 0 sur petites phrases
def Blue(ref_tokens,pred_tokens ):
    smooth_fn = SmoothingFunction().method1
    bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
    return bleu_score

# -------------------------------
# 2) Calcul ROUGE (rouge1, rouge2, rougeL)
# -------------------------------
def Rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    return rouge_scores

# -------------------------------
# 3) Calcul BERTScore (précision, rappel, F1)
# -------------------------------
def BERT(prediction, reference):
    return bert_score([prediction], [reference], lang='fr', rescale_with_baseline=True)

"""
# -------------------------------
# 4) Calcul MoverScore (entre 0 et 1)
# -------------------------------
# MoverScore attend des listes de tokens en minuscules
pred_tokens_mover = pred_tokens
ref_tokens_mover = ref_tokens

mover_score = word_mover_score([pred_tokens_mover], [ref_tokens_mover], lang='fr')
print(f"MoverScore: {mover_score[0]:.4f}")
"""

#############################################################################################

name = np.array(["BLEU score", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore Precision", "BERTScore Recall", "BERTScore F1"])
score = np.array([])

#phrase à remplir pour les tests 
references = ["La capitale de la France est Paris.", "Je joue au ping-pong", "J'aime manger une chocolatine", "Je pense donc je suis", "Bonjour comment ca va ?"]
predictions = ["Paris est la capitale de la France.", "Je joue au tennis de table", "J'adore manger un pain au chocolat", "La vie est un long fleuve tranquille", "Bonjour, bonjour, bonjour, bonjour"] 

for i in range(len(references)): 
    reference = references[i]
    prediction = predictions[i]

    score = np.array([])

    ref_tokens = preprocess(reference)
    pred_tokens = preprocess(prediction)

    print(f"Phrase numéro {i+1} : {prediction}")
    bleu_score = Blue(ref_tokens, pred_tokens)
    print(f"BLEU score: {bleu_score:.4f}")

    score = np.append(score, [bleu_score])

    rouge_scores = Rouge(reference, prediction)
    print(f"ROUGE-1 F1: {rouge_scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rougeL'].fmeasure:.4f}")
    score = np.append(score,[rouge_scores['rougeL'].fmeasure, rouge_scores['rouge2'].fmeasure, rouge_scores['rougeL'].fmeasure])


    P, R, F1 = BERT(prediction, reference)
    print(f"BERTScore Precision: {P.mean().item():.4f}")
    print(f"BERTScore Recall: {R.mean().item():.4f}")
    print(f"BERTScore F1: {F1.mean().item():.4f}")

    score = np.append(score,[P.mean().item(),R.mean().item(),F1.mean().item()])

    data_metrics = pd.DataFrame({"name": name, "score": score})

    sns.barplot(data=data_metrics, x='name', y='score', palette="pastel")
    plt.title(f"phrase n° {i+1} : {prediction} / {reference}")
    plt.show()

    #Bleu => mots clé présent sans changement 


    """
    BertScore > des autres métriques (remarquant mieux synonyme)
    """