# Medical Chatbot with RAG Pipeline

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for a **Medical Chatbot** using **FAISS** for vector search and **CTransformers** for language modeling. The chatbot retrieves information from a set of medical PDFs stored in a vector database and provides relevant answers to user queries.

## 🚀 Features

- **FAISS Vector Database**: Efficient similarity search on medical documents.
- **RAG Pipeline**: Enhances chatbot responses with document retrieval.
- **CTransformers LLM**: Uses Llama-2-7B-Chat-GGML model for generating responses.
- **Streamlit UI**: User-friendly web interface for interacting with the chatbot.
- **Chat History**: Logs user queries and responses in JSON format.
- **Evaluation Pipeline**: Uses ROUGE, BERTScore, and Semantic Similarity for performance assessment.

##  Results
![Screenshot from 2025-03-28 14-38-24](https://github.com/user-attachments/assets/322dfa73-1fa8-4749-8daf-c6c67970c84e)


## 🏗️ Project Structure
```
Final_Project/
│── data/                # Medical PDFs
│── vectorstore/dbfaiss         # FAISS vector database
│── medi_bot/
│   ├── ingest.py        # Creates FAISS vector database
│   ├── medical.py       # Chatbot implementation (Streamlit UI)
│   ├── evaluate.py      # Evaluates chatbot responses
│   ├── chat_history.json # Stores chat history
│   ├── medical_qa.csv   # Contains sample questions & answers
│   ├── results.csv      # Stores evaluation results
│── README.md              # Documentation
│── Requirements.txt         
```

## 🛠️ Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-repo/medical-chatbot.git
cd medical-chatbot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up FAISS Database
```bash
python ingest.py
```

### 4️⃣ Run Medical Chatbot
```bash
streamlit run medical.py
```

### 5️⃣ Run Evaluation
```bash
python evaluate.py
```

## 🏃 Usage
- Enter a medical-related question in the chat UI.
- The chatbot retrieves relevant information from stored PDFs.
- If sources are found, they are displayed with the response.
- Chat history is logged in `chat_history.json`.

## 🛠️ Configuration
Modify paths in `medical.py`, `ingest.py`, and `evaluate.py` to match your directory structure:


## 📊 Evaluation Metrics
The chatbot performance is assessed using:
- **ROUGE Score**: Measures text overlap.
- **BERT Score**: Evaluates contextual similarity.
- **Semantic Similarity**: Uses SBERT model.



