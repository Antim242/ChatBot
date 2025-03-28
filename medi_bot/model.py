import streamlit as st
import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

# Define paths
DB_FAISS_PATH = '/home/shtlp_0191/Documents/Final_Project/vectorstore/dbfaiss'
CHAT_LOG_FILE = "/home/shtlp_0191/Documents/Final_Project/medi_bot/chat_history.json"


# Ensure JSON file exists
def initialize_json():
    if not os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "w") as f:
            json.dump([], f)

# Save chat history in JSON
def save_chat_json(query, answer):
    chat_data = {"query": query, "answer": answer}
    try:
        with open(CHAT_LOG_FILE, "r") as f:
            chat_history = json.load(f)
        chat_history.append(chat_data)
        with open(CHAT_LOG_FILE, "w") as f:
            json.dump(chat_history, f, indent=4)
    except Exception as e:
        print(f"Error saving chat: {e}")

# Load the model
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

# Create custom prompt
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# QA Model
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )

# Streamlit UI
st.title("💬 Medical Chatbot")
st.write("Ask me any medical-related question!")

initialize_json()
qa_chain = qa_bot()

user_query = st.text_input("Enter your query:")

if st.button("Ask"):
    if user_query:
        res = qa_chain(user_query)
        answer = res["result"]
        sources = res["source_documents"]

        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(str(src) for src in sources)
        else:
            answer += "\n\n_No sources found._"

        # Save chat
        save_chat_json(user_query, answer)

        # Display response
        st.write(answer)
