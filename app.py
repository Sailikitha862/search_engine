# app.py
import streamlit as st
from search_engine import get_bot_response

st.title("🔍 Semantic Search Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question based on the documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = get_bot_response(prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
                response = "Something went wrong. Check console logs."
    st.session_state.messages.append({"role": "assistant", "content": response})
