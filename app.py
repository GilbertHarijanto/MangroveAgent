from agent import run_mangrove_agent
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
import shelve
import time

load_dotenv(find_dotenv(), override=True)

st.image('images/img.png')
st.subheader(
    'Real-time insights on mangrove health 🌱')

with st.sidebar:
    api_key = st.text_input('OpenAI API Key', type='password')
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()


USER_AVATAR = "👤"
BOT_AVATAR = "🤖"


# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):

        placeholder = st.empty()
        with st.spinner("Thinking..."):
            full_response = run_mangrove_agent(prompt, api_key)
            streamed = ""
            for char in full_response:
                streamed += char
                placeholder.markdown(streamed + "▌")
                time.sleep(0.005)
            placeholder.markdown(streamed)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
