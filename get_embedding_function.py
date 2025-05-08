from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from config import *

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return embeddings
