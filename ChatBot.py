"""
Code de William pour accéder au LLM sur AWS
"""

from langchain_aws import ChatBedrock
import boto3
import os


def ChatbotLLM(llm_model_name: str = None) -> ChatBedrock:

    """
    Chooses the appropriate Bedrock chat model based on user input.
    Returns:
        The initialized ChatBedrock model.
    """

    llm_model_name = llm_model_name or "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
        aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
    )

    boto3_bedrock = session.client(
        service_name="bedrock-runtime", region_name="us-east-1"
    )

    llm_model = ChatBedrock(

        model_id=llm_model_name,
        model_kwargs={"temperature": 0.3},
        client=boto3_bedrock,
    )

    return llm_model