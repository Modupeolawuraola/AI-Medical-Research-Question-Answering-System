from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
