import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

def setup_chroma():
    try:
        import tensorflow as tf
        client = chromadb.PersistentClient(path="./medical_research_db")
        try:
            return client.get_collection(
                name="medical_research",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
        except Exception:
            return client.create_collection(
                name="medical_research",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
    except Exception as e:
        print(f"Error in ChromaDB setup: {e}")
        raise

def store_documents(collection, documents, metadata):
    try:
        if collection.count() > 0:
            return
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=[f"doc_{j}" for j in range(i, i + len(batch_docs))]
            )
    except Exception as e:
        print(f"Error storing documents: {e}")
        raise


