
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_dataset(df):
    relevant_columns=['title', 'abstract', 'authors', 'publish_time', 'journal']
    df_cleaned = df[relevant_columns]
    df_cleaned=df_cleaned.dropna(subset=['abstract'])
    df_cleaned['text']="Title: " + df_cleaned['title'] + "\nAbstract: " + df_cleaned['abstract']
    return df_cleaned

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
