"""
Streamlit main application entrypoint.
=== FILE: frontend/app.py ===
"""

import os
import streamlit as st
import importlib.util
import sys

# Add the project root to sys.path so we can import frontend components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat

# App UI config
st.set_page_config(
    page_title="RAG MLOps Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("Enterprise RAG Q&A System")
st.markdown("Ask questions against your ingested documents. Powered by FastAPI, LangChain, and MLflow.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    # Render the sidebar for document upload & settings
    render_sidebar()
    
    # Render the main chat interface
    render_chat()

if __name__ == "__main__":
    main()
