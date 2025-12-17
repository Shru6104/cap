# app.py (UPDATED – Multi-Part FAQ + Multi-Product Recommendations)

import streamlit as st
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
import os
import re

from utils.recommendation_engine import get_recommendation
from utils.faq_engine import get_faq_answer
from utils.detect import detect_module

# -------------------------------
# DATABASE
# -------------------------------
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat(sender, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat (sender, message, timestamp) VALUES (?, ?, ?)",
        (sender, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

init_db()

# -------------------------------
# LOAD DATASETS
# -------------------------------
df_customers = pd.read_csv("data/recommendation_dataset.csv")

# Normalize DOB as STRING
df_customers["CustomerDOB"] = pd.to_datetime(
    df_customers["CustomerDOB"],
    errors="coerce"
).dt.strftime("%d-%m-%Y")

# -------------------------------
# LOAD FAQ MODELS
# -------------------------------
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
df_faq = pd.read_excel("data/FAQ_dataset.xlsx")

def chatbot_faq(text):
    """Return FAQ answer if confidence > 0.4"""
    try:
        vec = vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        if max(probs) < 0.4:
            return None
        pred = model.predict(vec)[0]
        intent = label_encoder.inverse_transform([pred])[0]
        answers = df_faq[df_faq["intent"] == intent]["answer"]
        return answers.sample(1).values[0] if not answers.empty else None
    except:
        return None

def detect_faq_multi(user_input):
    """Split input by common separators and check FAQ for each segment"""
    segments = re.split(r'and|,', user_input, flags=re.IGNORECASE)
    faq_responses = []
    for seg in segments:
        ans = chatbot_faq(seg.strip())
        if ans:
            faq_responses.append(ans)
    return "\n".join(faq_responses) if faq_responses else None

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = "guest"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# LOGIN SCREEN
# -------------------------------
if not st.session_state.logged_in:
    st.title("🔐 Login")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Continue as Guest"):
            st.session_state.logged_in = True
            st.session_state.role = "guest"
            st.rerun()

    with col2:
        cust_id = st.text_input("Customer ID (e.g. C5841053)")
        dob_input = st.text_input("Date of Birth (DD-MM-YYYY)")

        if st.button("Login"):
            record = df_customers[
                (df_customers["CustomerID"] == cust_id) &
                (df_customers["CustomerDOB"] == dob_input)
            ]

            if not record.empty:
                st.session_state.logged_in = True
                st.session_state.role = "customer"
                st.session_state.customer_id = cust_id
                st.rerun()
            else:
                st.error("Invalid Customer ID or DOB")

    st.stop()

# -------------------------------
# CHAT UI
# -------------------------------
st.set_page_config(page_title="Banking Chatbot", layout="centered")
st.title("🏦 Banking Assistant Chatbot")

# LOGOUT
if st.button("Logout"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# INPUT
user_input = st.text_input("Enter your message")

if st.button("Send") and user_input.strip():
    final_responses = []

    # -----------------------
    # Multi-part FAQ detection
    # -----------------------
    faq_ans = detect_faq_multi(user_input)
    if faq_ans:
        final_responses.append(faq_ans)  # Clean output, no label

    # -----------------------
    # Detect recommendations
    # -----------------------
    parts = detect_module(user_input)
    if parts.get("recommendation"):
        if st.session_state.role == "customer":
            rec_ans = get_recommendation(
                parts["recommendation"], 
                logged_in_customer_id=st.session_state.customer_id
            )
            rec_ans = rec_ans.replace("\n", "  \n")  # Markdown bullets
            final_responses.append(rec_ans)
        else:
            final_responses.append("🔒 Please login as customer to get recommendations.")

    # -----------------------
    # Fallback if nothing detected
    # -----------------------
    if not final_responses:
        final_responses.append("Sorry, I couldn't understand that. Please ask a banking-related question.")

    # Combine all responses
    final_reply = "\n\n".join(final_responses)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", final_reply))

    save_chat("You", user_input)
    save_chat("Bot", final_reply)

# -------------------------------
# DISPLAY CHAT
# -------------------------------
st.write("### 💬 Conversation")

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💼 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
