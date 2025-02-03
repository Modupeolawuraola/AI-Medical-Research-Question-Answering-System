FROM ollama/ollama

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the entire project
COPY . .


# Create a startup script with localhost instead of 0.0.0.0
RUN echo '#!/bin/sh\nollama serve & sleep 10 && streamlit run src/rag/ui/streamlit_app.py --browser.serverAddress=localhost' > /app/start.sh
RUN chmod +x /app/start.sh

# Expose port for Streamlit
EXPOSE 8501

# Run the startup script
ENTRYPOINT ["/app/start.sh"]