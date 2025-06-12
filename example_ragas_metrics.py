from ragas import SingleTurnSample
from ragas.metrics import BleuScore
"""
##Sans appel a des LLM 

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
}
metric = BleuScore()
test_data = SingleTurnSample(**test_data)
print(metric.single_turn_score(test_data))
"""
#####################################################################################################################
##Avec appel LLM 

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

import pandas as pd

def generate_answer(question, contexts):
    prompt = (
        "Answer the user question **only** with facts found in the context.\n\n"
        "Context:\n"
        + f"{contexts}"
        + f"\n\nQuestion: {question}\nAnswer:"
    )

    print(prompt)

    response = llm.invoke(prompt)

    return response.content.strip()


llm = ChatOllama(model="gemma3:1b", temperature=0)
ragas_llm = LangchainLLMWrapper(llm)

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ragas_emb = LangchainEmbeddingsWrapper(emb)


questions = [
    "What's the club's name?",
    #"When and who founded us métro?",
    #"Who's the captain of this team?",
]

ground_truths = [
    "US metro",
    #"Jean Yves in 1998",
    #"Arnaud"
]

contexts = [
    "The name of the soccer club is US Metro.", 
    #"It was founded in 1998 by Jean Yves after he left the PSG.", 
    #"Young Arnaud has been club captain for 5 months."
]
rows = []

for question, ground_truth, context in zip(questions, ground_truths, contexts):

    answer = generate_answer(question, context)
    rows.append(
        {
            "question": question,
            "contexts": [context],
            "answer": answer,
            "reference": ground_truth,
        }
    )

evaluation_dataset = Dataset.from_list(rows)
print("tab : ", rows)


scores = evaluate(
    evaluation_dataset,
    metrics=[answer_correctness, answer_relevancy, faithfulness,
             context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_emb,
)

# Fusionner les scores avec les données initiales
score_dict = scores.to_pandas().to_dict(orient="records")

# Construire une liste de lignes combinées
export_rows = []

for row, score in zip(rows, score_dict):
    export_rows.append({
        "Question": row["question"],
        "Contexte": row["contexts"][0],
        "Référence attendue": row["reference"],
        "Réponse générée": row["answer"],
        "Score - Correctness": score["answer_correctness"],
        "Score - Relevancy": score["answer_relevancy"],
        "Score - Faithfulness": score["faithfulness"],
        "Score - Context Precision": score["context_precision"],
        "Score - Context Recall": score["context_recall"],
    })

# Créer le DataFrame
df = pd.DataFrame(export_rows)

# Sauvegarder en Excel
df.to_excel("resultats_ragas_evaluation.xlsx", index=False)

print("✅ Résultats enregistrés dans 'resultats_ragas_evaluation.xlsx'")
