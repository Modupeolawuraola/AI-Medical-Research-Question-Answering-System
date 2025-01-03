# AI-Medical-Research-Question-Answering-System
AI - Medical QA System 

A RAG (Retrieval-Augmented Generation) system that utilizes Ollama LLM to answer medical research queries with source citations.

## Features
- Vector-based search using ChromaDB 
- Natural language query interface with Streamlit
- Source citation for answers
- Semantic matching using Sentence Transformers

## Prerequisites
- Python 3.8+
- Ollama (install from [ollama.ai](https://ollama.ai))
- 8GB RAM minimum
- 20GB disk space

## Installation
```bash
# Clone repo
git clone https://github.com/Modupeolawuraola/AI-Medical-Research-Question-Answering-System

# Install dependencies
pip install -r requirements.txt

# Pull Llama2 model
ollama pull llama2

#usage
streamlit run Q_and_A_medical_dignosis_RAG_streamlit.py
```
## App Screenshot 
