# Usage instruction

## Step1: clone the repository by using this command

```bash
git clone https://github.com/Modupeolawuraola/AI-Medical-Research-Question-Answering-System
````
## Step 2: Create a virtual environment and setup the virtual environment with the command below

```bash
python -m venv -rag_env
source rag_env/bin/activate
```
# Step3: Install dependencies
```bash 
pip install -r requirements.txt
```

## Step4 : download ollama locally on your computer  using the link below

```bash
install ollama : https://ollama.com/download/windows
```

# Step5: Pull Llama2 model
```bash 
ollama pull llama2
```

## Step6: re-run the setuptool -setup.py and pyproject.toml tool by running the command below

```bash
pip3 install  -e .
```

## step7: Run the test code to confirm the app is running effectively with the command below

```bash
python test_rag.py
```

## Step8: Run the streamlit appy (AI Research assistant)

```bash
streamlit run streamlit_app.py
```
