from ragas import SingleTurnSample
from ragas.metrics import BleuScore
"""
#----------------------------------------------------------------------------------------
#Sans appel a des LLM (BLEU SCORE EXEMPLE)
#----------------------------------------------------------------------------------------

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
}
metric = BleuScore()
test_data = SingleTurnSample(**test_data)
print(metric.single_turn_score(test_data))
"""
#----------------------------------------------------------------------------------------
#Avec appel LLM 
#----------------------------------------------------------------------------------------

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama

from ragas import evaluate, RunConfig
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

import pandas as pd

def generate_answer(llm,question, contexts):
    prompt = (
        "Answer the user question **only** with facts found in the context.\n\n"
        "Context:\n"
        + f"{contexts}"
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    #print(prompt)
    response = llm.invoke(prompt)

    return response.content.strip()

def load_csv_data():
    
    data = pd.read_excel("data/input/data.xlsx")

    # Extraire les colonnes en listes
    questions = data["question"].tolist()
    contexts = data["context"].tolist()
    ground_truths = data["reference"].tolist()

    return questions, contexts, ground_truths



def get_dataset(llm, questions, ground_truths, contexts, rows):

    for question, ground_truth, context in zip(questions, ground_truths, contexts):

        answer = generate_answer(llm, question, context)
        rows.append(
            {
                "question": question,
                "contexts": [context],
                "answer": answer,
                "reference": ground_truth,
            }
        )
        print("-------------------------------------------------")
        print(f"- Question : {question} \n - Response LLM : {answer}")
        print("------------------------------------------------- \n \n")

    return Dataset.from_list(rows)


def Evaluations(ragas_llm, ragas_emb,evaluation_dataset):

    my_config = RunConfig(
    max_workers=1,
    )

    scores = evaluate(
    evaluation_dataset,
    metrics=[answer_correctness, answer_relevancy, faithfulness,
             context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_emb,
    run_config=my_config
)
    return scores


def create_data_frame(rows, score_dict):
    # Construire une liste de lignes combinées
    export_rows = []

    for row, score in zip(rows, score_dict):
        export_rows.append({
            "Question": row["question"],
            "Contexte": row["contexts"][0],
            "Réponse attendue": row["reference"],
            "Réponse générée": row["answer"],
            "Correctness": score["answer_correctness"],
            "Relevancy": score["answer_relevancy"],
            "Faithfulness": score["faithfulness"],
            "Context Precision": score["context_precision"],
            "Context Recall": score["context_recall"],
        })

    # Créer le DataFrame
    return pd.DataFrame(export_rows)


#---------------------------------------------------------------------------------------
def main():
    #création des modèle
    llm = ChatOllama(model="gemma3:1b", temperature=0, timeout=120) #changer le nom du modèle 
    ragas_llm = LangchainLLMWrapper(llm)

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    #load question, contexte, réponse depuis fichier xlsx (data/input)
    questions, contexts, ground_truths = load_csv_data()

    rows = []

    #création et évaluation de notre dataset
    evaluation_dataset = get_dataset(llm, questions, ground_truths, contexts, rows) #création du dataset
    scores = Evaluations(ragas_llm, ragas_emb, evaluation_dataset) #évaluation de notre dataset

    #affichage dans un exel
    # Fusionner les scores avec les données initiales
    score_dict = scores.to_pandas().to_dict(orient="records")

    df = create_data_frame(rows, score_dict)
    # Sauvegarder en Excel
    df.to_excel("data/output/resultats_ragas_evaluation.xlsx", index=False)

    print("✅ Résultats enregistrés dans 'data/output/resultats_ragas_evaluation.xlsx'")

if __name__ == "__main__":
    main()