import pandas as pd

data = pd.read_excel("data/input/data.xlsx")

# Extraire les colonnes en listes
questions = data["question"].tolist()
contexts = data["context"].tolist()
ground_truths = data["reference"].tolist()

print(questions, contexts, ground_truths)