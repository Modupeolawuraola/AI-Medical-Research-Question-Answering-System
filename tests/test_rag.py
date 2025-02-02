from rag.utils import process_metadata
from rag.preprocessing import clean_dataset, preprocess_text
from rag.database import setup_chroma
from rag.retrieval import setup_ollama, medical_qa

def test_pipeline():
    # Test data loading
    df = process_metadata()
    assert df is not None, "Data loading failed"
    
    # Test preprocessing
    df_cleaned = clean_dataset(df)
    documents, metadata = preprocess_text(df_cleaned)
    assert len(documents) > 0, "Text preprocessing failed"
    
    # Test ChromaDB
    collection = setup_chroma()
    assert collection is not None, "ChromaDB setup failed"
    
    # Test Ollama
    ollama_query = setup_ollama()
    assert ollama_query is not None, "Ollama setup failed"
    
    # Test query
    response = medical_qa("What are common COVID symptoms?", collection, ollama_query)
    assert response["answer"] is not None, "Query failed"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
