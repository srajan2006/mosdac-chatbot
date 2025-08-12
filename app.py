import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import google.generativeai as genai

# ---- Load API Key from Streamlit secrets ----
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# ---- Load Knowledge Base ----
df = pd.read_csv("data.csv")

# ---- Create Embeddings ----
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['content'].tolist())

# ---- Create FAISS index ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ---- Helper: Search ----
def retrieve_answer(query, top_k=2):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    context = "\n".join(df.iloc[idx]['content'] for idx in indices[0])
    return context

# ---- Helper: Ask Gemini ----
def ask_gemini(question, context):
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
    model_g = genai.GenerativeModel("gemini-pro")
    response = model_g.generate_content(prompt)
    return response.text

# ---- UI ----
st.set_page_config(page_title="MOSDAC Query Chatbot", page_icon="🛰️")

st.title("🛰️ MOSDAC Query Chatbot")
st.write("Ask me about satellites or MOSDAC data access.")

# Login / Guest mode
mode = st.radio("Choose mode:", ["Guest", "Login"])
if mode == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username and password:
        st.success(f"Logged in as {username}")
else:
    st.info("You are in Guest mode.")

# Chat input
user_q = st.text_input("Your question:")
if st.button("Ask"):
    if user_q.strip():
        context = retrieve_answer(user_q)
        answer = ask_gemini(user_q, context)
        st.write("**Answer:**", answer)
        st.caption("Sources: Based on MOSDAC and satellite facts dataset.")
