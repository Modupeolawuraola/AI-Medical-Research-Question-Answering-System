import streamlit as st
from rag.preprocessing import clean_dataset, preprocess_text
from rag.database import setup_chroma, store_documents
from rag.utils import process_metadata
from rag.retrieval import setup_ollama, medical_qa


def initialize_system():
    """Initialize the RAG system with ChromaDB and Ollama"""
    try:
        with st.spinner('Loading the medical research system...'):
            # Check if collection already exists in session state
            if 'collection' not in st.session_state:
                # Process data
                st.info("Processing metadata...")
                df = process_metadata()
                if df is None:
                    st.error("Failed to load dataset. Check if data files exist.")
                    return False

                st.info("Cleaning dataset...")
                df_cleaned = clean_dataset(df)
                documents, metadata = preprocess_text(df_cleaned)

                st.info("Setting up ChromaDB...")
                try:
                    collection = setup_chroma()
                    if collection is not None:
                        st.info("Storing documents in ChromaDB...")
                        store_documents(collection, documents, metadata)
                        st.session_state.collection = collection
                        st.success("Database setup complete!")
                        return True
                    else:
                        st.error("ChromaDB setup failed")
                        return False
                except Exception as e:
                    st.error(f"ChromaDB Error: {str(e)}")
                    return False
            return True  # If collection already exists
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return False


def init_ollama():
    """Initialize Ollama if not already done"""
    try:
        if 'ollama_query' not in st.session_state:
            with st.spinner('Setting up the AI model...'):
                st.info("Connecting to Ollama...")
                ollama_query = setup_ollama("llama2")
                if ollama_query is not None:
                    st.session_state.ollama_query = ollama_query
                    st.success("AI Model connected successfully!")
                    return True
                else:
                    st.error("Failed to connect to Ollama. Make sure Ollama is running.")
                    return False
        return True
    except Exception as e:
        st.error(f"Ollama Error: {str(e)}")
        return False


def main():
    st.title("Medical Research Question Answering System")
    # updated main description
    st.markdown(""" ### 🏥 AI Medical Research Assistant
    This AI system  analyzes medical research papers to answer questions about medical topics, 
    diagnosis  with particular expertise in respiratory diseases and infections. 
    Ask a question about:
    - General medical conditions and treatments
    - Research Findings and clinical studies
    - Medical procedures and protocols
    - Disease symptoms and diagnoses 
    - And more.......

    While the system can handle general medical queries, it has extensive knowledge about respiratory health 
    from specialized research papers in this field. The system will search through research papers to provide relevant answers with citations.
    """)

    # Initialize system components
    system_ready = initialize_system()
    ollama_ready = init_ollama()

    if not system_ready or not ollama_ready:
        st.error("System initialization failed. Please check the logs above and try again.")
        return

    # Input section
    query = st.text_area(
        "Enter your medical research question:",
        height=100,
        placeholder="""Example: 
                    "What are the main symptoms of respiratory infections?"
                    "What research exist on asthma management"
                    "What are common complications of viral infections"""
    )

    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching through research papers..."):
            try:
                response = medical_qa(
                    query,
                    st.session_state.collection,
                    st.session_state.ollama_query
                )

                if response and isinstance(response, dict):
                    # Display answer
                    st.markdown("### Answer")
                    st.write(response["answer"])

                    # Display sources
                    st.markdown("### 📚Research Sources")
                    for source in response["sources"]:  # Fixed: 'sources' instead of 'source'
                        st.markdown(f"""
                        📚Title: **{source['title']}
                        ** 📅 Published: {source['publish_time']}
                        """)
                else:
                    st.error("Failed to get a valid response from the system.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.markdown("### About- AI-Medical Research Question-Answering System")
        st.markdown("""
        This advanced medical research assistant uses:
        - AI-powered analysis of medical literature
        - Comprehensive medical research database
        - Specialized focus on respiratory medicine
        - Real-time research paper analysis

        #### 🎯 Research Coverage
        - Primary focus: Respiratory diseases
        - Secondary: General medical topics
        - Research papers from peer-reviewed journals

        #### 💡 Tips for Best Results
        1. Be specific in your questions
        2. Include relevant medical terms
        3. Specify the aspect you're most interested in
        4. Wait for thorough analysis of sources
        """)

        # System status indicators
        st.markdown("### System Status")
        st.markdown("✅ Database: Connected" if system_ready else "❌ Database: Not Connected")
        st.markdown("✅ AI Model: Ready" if ollama_ready else "❌ AI Model: Not Ready")


if __name__ == "__main__":
    main()