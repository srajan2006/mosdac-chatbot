# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------- Load Environment Variables --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------- Load Knowledge Base --------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data("data.csv")

# -------------------- Create TF-IDF Matrix --------------------
@st.cache_resource
def create_vectorizer(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['content'].tolist())
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_vectorizer(df)

# -------------------- Retrieve Relevant Context --------------------
def retrieve_answer(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray()
    ranked_indices = np.argsort(scores[:, 0])[::-1][:top_k]
    context = "\n".join(df.iloc[idx]['content'] for idx in ranked_indices)
    return context

# -------------------- Ask Gemini API --------------------
def ask_gemini(question, context):
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
    model_g = genai.GenerativeModel("gemini-pro")
    response = model_g.generate_content(prompt)
    return response.text.strip()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="MOSDAC Query Chatbot", page_icon="🛰️")
st.title("🛰️ MOSDAC Query Chatbot")
st.write("Ask me about satellites or MOSDAC data access.")

# Login / Guest mode
mode = st.radio("Choose mode:", ["Guest", "Login"])

if mode == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username and password:
        st.success(f"✅ Logged in as {username}")
else:
    st.info("You are in Guest mode.")

# Chat input
user_q = st.text_input("Your question:")

if st.button("Ask"):
    if user_q.strip():
        context = retrieve_answer(user_q)
        answer = ask_gemini(user_q, context)
        st.markdown(f"**Answer:** {answer}")
        st.caption("📚 Sources: Based on MOSDAC and satellite facts dataset.")
