from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

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

from ChatBot import ChatbotLLM
"""
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
"""

def load_csv_data(path_doc):
    
    data = pd.read_excel(path_doc)

    # Extraire les colonnes en listes
    questions = data["question"].tolist()
    contexts = data["context"].tolist()
    ground_truths = data["reference"].tolist()

    #valable si on a déjà la réponse de LLM
    answer = data["answer"].tolist()

    return questions, contexts, ground_truths, answer



def get_dataset(llm, questions, ground_truths, contexts,answers, rows):

    for question, ground_truth, context, answer in zip(questions, ground_truths, contexts, answers):

        #answer = generate_answer(llm, question, context)
        #pour éviter de fair trop d'appel couteux
    
        rows.append(
            {
                "question": question,
                "contexts": [context],
                "answer": answer,
                "reference": ground_truth,
            }
        )
        print("-------------------------------------------------")
        print(f"- Question : {question} \n - Response LLM (recup dans le dataset) : {answer}")
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
    llm = ChatbotLLM()
    ragas_llm = LangchainLLMWrapper(llm)

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    #------------------------
    path_input_doc = "data/input/data_V2.xlsx"
    path_output_doc = "data/output/resultats_ragas_evaluation_V2.xlsx"
    #------------------------



    #load question, contexte, réponse depuis fichier xlsx (data/input)
    questions, contexts, ground_truths, answer = load_csv_data(path_input_doc)

    rows = []

    #création et évaluation de notre dataset
    evaluation_dataset = get_dataset(llm, questions, ground_truths, contexts, answer,  rows) #création du dataset
    scores = Evaluations(ragas_llm, ragas_emb, evaluation_dataset) #évaluation de notre dataset

    #affichage dans un exel
    # Fusionner les scores avec les données initiales
    score_dict = scores.to_pandas().to_dict(orient="records")

    df = create_data_frame(rows, score_dict)
    # Sauvegarder en Excel
    df.to_excel(path_output_doc, index=False)

    print(f"✅ Résultats enregistrés dans {path_output_doc}")

if __name__ == "__main__":
    main()