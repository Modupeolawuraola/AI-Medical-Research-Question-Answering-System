#%%

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
import zipfile
import requests
import json
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional, Dict, Any




def setup_kaggle():
    try:
        api = KaggleApi()
        api.authenticate()
        print('Kaggle API authenticated')
        return api
    except Exception as e:
        print(f'Authentication Error: {e}')
        return None


def process_metadata():
    # Check if the unzipped file already exists
    if os.path.exists('./data/covid19_subset.csv'):  # Check this path
        print('Loading existing processed data...')
        try:
            df = pd.read_csv('./data/covid19_subset.csv')
            print("Data loaded successfully")
            return df
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return None

    #checking if zip file exists
    if os.path.exists('./data/metadata.zip'):
        print('Metadata found; extracting file starting.....')
        with zipfile.ZipFile('/data/metadata.zip', 'r') as zip_re:
            zip_re.extractall('./data')

    else:
        api=setup_kaggle()
        if api:
            try:
                api.dataset_download_file(
                    'allen-institute-for-ai/CORD-19-research-challenge',
                    'metadata.csv',
                    path='./data')
                print('Metadata downloaded Successfully')

                #extract the zip file
                with zipfile.ZipFile('./data/metadata.csv.zip', 'r') as zip_ref:
                    zip_ref.extractall('./data')
            except Exception as e:
                print(f"Download Error: {e}")
                return None


    try:
        #processing the extracted dataset
        print("Dataset Processing")
        chunks = pd.read_csv('./data/metadata.csv.zip', chunksize= 1000)
        selected_data=pd.concat([next(chunks) for _ in range(5)])

        #save processed data
        selected_data.to_csv('./data/covid19_subset.csv', index=False)
        print("Created smaller subset of data!")

        #set display options to show all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)

        #Preview the dataset
        print("\nPreview of the data:")
        print(selected_data.head().to_string())

        #display column information '
        print("\nDetailed column Information")
        for col in selected_data.columns:
            non_null = selected_data[col].count()
            dtype =selected_data[col].dtype
            print(f"{col:20} | Type: {dtype :10} | Non-null count:{non_null}")

        #display basic statistics
        print("\nBasic Statistics::")
        print(selected_data.describe().to_string())

        #display sample of text columns (like abstract and title)
        print("\nSample Title:")
        print(selected_data['title'].head().to_string())

        print("\nSample Abstract:")
        print(selected_data['abstract'].head().to_string())

        return selected_data


    except Exception as e:
        print(f"Processing Error: {e}")
        return None



#RAG Implementation fro Questioning and answering system about medical research
def clean_dataset(df):
    relevant_columns=['title', 'abstract', 'authors', 'publish_time', 'journal']
    df_cleaned = df[relevant_columns]

    #remove missing values in abstract
    df_cleaned=df_cleaned.dropna(subset=['abstract'])

    df_cleaned['text']="Title: " + df_cleaned['title'] + "\nAbstract: " + df_cleaned['abstract']

    return df_cleaned

from langchain.text_splitter import RecursiveCharacterTextSplitter

def preprocess_text(df):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]

    )

    documents=[]
    metadata=[]

    for idx, row in df.iterrows():
        chunks =text_splitter.split_text(row['text'])

        for chunk in chunks:
            documents.append(chunk)
            metadata.append({
                'title':row['title'],
                'authors': row['authors'],
                'publish_time': row['publish_time'],
            })

    return documents, metadata

#setup chromadb and store embeddings
import chromadb
from chromadb.utils import embedding_functions


def setup_chroma():
    """Setup ChromaDB with minimal output"""
    try:
        import tensorflow as tf
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path="./medical_research_db")

        try:
            existing_collection = client.get_collection(
                name="medical_research",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
            return existing_collection

        except Exception:
            # Create new collection only if it doesn't exist
            collection = client.create_collection(
                name="medical_research",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
            return collection

    except Exception as e:
        print(f"Error in ChromaDB setup: {e}")
        raise
#store documents - creating vector stores
def store_documents(collection, documents, metadata):
    """Store documents quietly"""
    try:
        # Get existing document count
        existing_count = collection.count()
        if existing_count > 0:
            return

        # Store in batches only if collection is empty
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

#testing hugging face access token
def test_hf_token():
    try:
        from huggingface_hub import HfApi
        api=HfApi()
        identity = api.whoami()
        print(f"Successfully authenticated as :{identity}")
        return True
    except Exception as e:
        print(f"Token verification failed: {e}")
        return False

#query and retrival functions
def setup_ollama(model_name="llama2"):
    """
    Setup Ollama model for inference
    using Huggingface API directly
    """

    def query_ollama(prompt, temperature=0.5):
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         "model": model_name,
                                         "prompt": prompt,
                                         "temperature": temperature,
                                         "stream": False
                                     }
                                     )

            if response.status_code == 200:
                result = response.json()['response']
                print("Ollama test successful!")
                print("Response:", result)
                return result
            else:
                print(f"Error: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return None

    # Test the connection
    try:
        test_response = query_ollama("Test connection")
        if test_response:
            return query_ollama
        return None
    except Exception as e:
        print(f"Error setting up Ollama: {e}")
        return None

def query_medical_research(collection, query, n_results=3):
    # Search for relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def medical_qa(query, collection, ollama_query):
    """
    Medical QA using Ollama and ChromaDB
    """
    try:
        # Get relevant contexts
        results = collection.query(
            query_texts=[query],
            n_results=3
        )

        if not results or not results['documents']:
            return {"answer": "No relevant documents found", "sources": []}

        # Format context
        context = "\n".join(results['documents'][0])

        # Create prompt
        prompt = f"""Based on the following medical research context, please provide a detailed answer.
        Focus on medical findings and cite specific research evidence when possible.

        Context: {context}

        Question: {query}

        Answer:"""

        # Get response from Ollama
        response = ollama_query(prompt)

        # Add source information
        sources = [
            {
                "title": meta["title"],
                "publish_time": meta["publish_time"]
            }
            for meta in results["metadatas"][0]
        ]

        return {
            "answer": response,
            "sources": sources
        }

    except Exception as e:
        print(f"Error in medical QA: {e}")
        return {"answer": f"Error: {str(e)}", "sources": []}

#main execution


def main():
    try:
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)

        # Process metadata first
        print("Loading dataset...")
        df = process_metadata()
        if df is None:
            raise ValueError("Failed to load dataset")
        print("Dataset loaded successfully")

        # Clean dataset
        print("Cleaning dataset...")
        df_cleaned = clean_dataset(df)
        print("Dataset cleaned")

        # Preprocess and chunk text
        print("Preprocessing text...")
        documents, metadata = preprocess_text(df_cleaned)
        print(f"Created {len(documents)} chunks")

        # Setup ChromaDB
        print("Setting up ChromaDB...")
        collection = setup_chroma()
        if collection is None:
            raise ValueError("ChromaDB setup failed")
        print("ChromaDB setup complete")

        # Store documents
        print("Storing documents...")
        if documents is None or metadata is None:
            raise ValueError("Documents or metadata is None")
        store_documents(collection, documents, metadata)
        print("Documents stored in ChromaDB")

        #setup LLM
        print("Setting up LLM.....")
        #Replace LLM setup with Ollama setup
        print("Setting up Ollama.....")
        ollama_query = setup_ollama("llama2")  # or another model you've pulled
        if ollama_query is None:
            raise ValueError("Ollama setup failed - check if Ollama is running")
        print("Ollama setup complete")

        # Example query
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




