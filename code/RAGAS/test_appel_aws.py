
from langchain_core.messages import HumanMessage
import os
from ChatBot import ChatbotLLM


#verifications de variable d'environnement
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("AWS_DEFAULT_REGION:", os.getenv("AWS_DEFAULT_REGION"))



llm = ChatbotLLM()

#Exemple de génération
"""
# Crée une question à envoyer au modèle
question = "Donne moi les meileurs joueurs qui ont joué pour les Girondins de Bordeaux"

# Envoie la question au LLM
response = llm.invoke([HumanMessage(content=question)])

# Affiche la réponse
print(response.content)
"""

"""
messages = [
    SystemMessage(content="Tu es un assistant expert en physique, donne des réponses claires et courtes."),
    HumanMessage(content="Qu'est-ce que la relativité ?")
]

response = llm.invoke(messages)
print(response.content)


"""