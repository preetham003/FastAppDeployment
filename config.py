import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
# load environment variables from the .env file 
from dotenv import load_dotenv
from azure.ai.inference import EmbeddingsClient
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI

load_dotenv()

# Set up Azure OpenAI Embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",  
    azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key= os.getenv("OPENAI_API_KEY"),
    azure_deployment="text-embedding-ada-002", 
    api_version="2024-02-15-preview"
)

azure_llm = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",  # Your deployment name
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o" # or "gpt-35-turbo" or the model you deployed
)

embeddings = EmbeddingsClient(
    endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("OPENAI_API_KEY"))
)

print(f"search index name: {os.environ['AISEARCH_INDEX_NAME']}")
# This client will be used to create and delete search indexes
search_client = SearchClient(
    index_name=os.environ["AISEARCH_INDEX_NAME"],
    endpoint=os.getenv("SEARCH_INDEX_ENDPOINT"),
    credential=AzureKeyCredential(key=os.getenv("SEARCH_INDEX_KEY")),
)