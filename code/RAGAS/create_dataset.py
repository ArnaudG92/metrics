"""
### Avec des PDF mais ca marche pas 

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragas.testset import TestsetGenerator



from langchain_ollama.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # Chargement PDF
    loader = PyMuPDFLoader("data/input/388948eng.pdf")
    documents = loader.load()

    # Découpage en chunks (ici on garde tous les chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(documents)
    # Pour tester, on peut limiter par exemple : chunks = chunks[155:355]

    # Modèle de génération de questions/réponses (data_generation_model)
    llm_gen = ChatOllama(model="gemma3:1b", temperature=0, timeout=120)
    ragas_llm_gen = LangchainLLMWrapper(llm_gen)

    # Modèle critique (critique la qualité des questions)
    llm_critic = ChatOllama(model="gemma3:1b", temperature=0, timeout=120)
    ragas_llm_critic = LangchainLLMWrapper(llm_critic)

    # Modèle d'embeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    # Création du générateur avec méthode from_langchain pour bien passer les 3 modèles
    generator = TestsetGenerator.from_langchain(
        data_generation_model=ragas_llm_gen,
        critic_model=ragas_llm_critic,
        embeddings=ragas_emb
    )

    # Distribution des types de questions que tu souhaites générer (modifiable)
 
    # Génération du dataset avec la distribution et taille souhaitée (ex: 10)
    dataset = generator.generate_with_langchain_docs(chunks, testset_size=10)

    print(dataset)
    # Tu peux aussi exporter au format DataFrame/CSV si besoin :
    df = dataset.to_pandas()
    df.to_csv('data/output/generated_dataset.csv', index=False)

if __name__ == "__main__":
    main()
"""

from langchain_community.document_loaders import DirectoryLoader

from langchain_ollama.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator

from ChatBot import ChatbotLLM

import csv

# from ragas.testset.evolutions import simple, reasoning, multi_context


path = "data/input"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

print(f"Documents chargés : {len(docs)}")
print(f"Extrait {docs[0]}")

llm = ChatOllama(model="gemma3:1b", temperature=0, timeout=120) #changer le nom du modèle 
#llm = ChatbotLLM()
#mistral:instruct
#gemma3:1b
ragas_llm = LangchainLLMWrapper(llm)


emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ragas_emb = LangchainEmbeddingsWrapper(emb)

print("\n Modèle chargé \n")

generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_emb, )
dataset = generator.generate_with_langchain_docs(docs, testset_size=7)

dataset = dataset.to_pandas()

dataset.to_csv("data/output/generated_dataset.csv", index=False, encoding="utf-8", sep=";", quoting=csv.QUOTE_ALL)


print("----------------------------------------------")
print("🆗 : dataset généré")
print("----------------------------------------------")


for idx, row in dataset.iterrows():
    print(f"🟢 Question {idx+1} : {row['user_input']}")
    #print(f"📚 Contexte : {row['reference_contexts']}")
    print(f"✅ Réponse : {row['reference']}")
    #print(f"🧠 Synthétiseur : {row['synthesizer_name']}\n")
    print("-" * 80)

