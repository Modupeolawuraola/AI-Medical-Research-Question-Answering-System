import os
from rag.utils import process_metadata
from rag.preprocessing import clean_dataset, preprocess_text
from rag.database import setup_chroma, store_documents
from rag.retrieval import setup_ollama, medical_qa

def main():
    try:
        os.makedirs('./data', exist_ok=True)
        
        print("Loading dataset...")
        df = process_metadata()
        if df is None:
            raise ValueError("Failed to load dataset")
            
        print("Cleaning dataset...")
        df_cleaned = clean_dataset(df)
        
        print("Preprocessing text...")
        documents, metadata = preprocess_text(df_cleaned)
        print(f"Created {len(documents)} chunks")
        
        print("Setting up ChromaDB...")
        collection = setup_chroma()
        if collection is None:
            raise ValueError("ChromaDB setup failed")
            
        print("Storing documents...")
        if documents is None or metadata is None:
            raise ValueError("Documents or metadata is None")
        store_documents(collection, documents, metadata)
        
        print("Setting up Ollama...")
        ollama_query = setup_ollama("llama2")
        if ollama_query is None:
            raise ValueError("Ollama setup failed - check if Ollama is running")
            
        query = "What are the main symptoms of respiratory infections in these studies?"
        print("\nQuery:", query)
        
        response = medical_qa(query, collection, ollama_query)
        print("\nAnswer:", response["answer"])
        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source['title']} ({source['publish_time']})")
            
        return response
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == '__main__':
    main()
