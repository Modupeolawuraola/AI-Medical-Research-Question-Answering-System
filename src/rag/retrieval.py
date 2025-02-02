import requests

def setup_ollama(model_name="llama2"):
    def query_ollama(prompt, temperature=0.5):
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         "model": model_name,
                                         "prompt": prompt,
                                         "temperature": temperature,
                                         "stream": False
                                     })
            if response.status_code == 200:
                return response.json()['response']
            return None
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return None

    try:
        test_response = query_ollama("Test connection")
        if test_response:
            return query_ollama
        return None
    except Exception as e:
        print(f"Error setting up Ollama: {e}")
        return None

def query_medical_research(collection, query, n_results=3):
    return collection.query(
        query_texts=[query],
        n_results=n_results
    )

def medical_qa(query, collection, ollama_query):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        if not results or not results['documents']:
            return {"answer": "No relevant documents found", "sources": []}
        context = "\n".join(results['documents'][0])
        prompt = f"""Based on the following medical research context, please provide a detailed answer.
        Focus on medical findings and cite specific research evidence when possible.
        Context: {context}
        Question: {query}
        Answer:"""
        response = ollama_query(prompt)
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
