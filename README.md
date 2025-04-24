## Local RAG Application Prototype


### Features
- upload pdf document and ask questions about pdf


### Models 
- Embedding model: all-MiniLM-L6-v2
- Language model: Phi-3

### How to run the application?

1. install the dependencies
```bash
pip install -r requirements.txt
```

2. download the required embedding model and language model
```bash
python download_models.py
```

3. run the applicaiton
```bash
streamlit run app.py
```