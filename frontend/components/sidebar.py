"""
Streamlit sidebar component for uploading inputs.
=== FILE: frontend/components/sidebar.py ===
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8002")

def render_sidebar():
    """
    Renders the sidebar panel allowing users to upload documents for ingestion.
    """
    with st.sidebar:
        st.header("Upload Document")
        st.write("Ingest new PDFs or TXTs into the knowledge base.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
        
        if st.button("Ingest"):
            if uploaded_file is not None:
                with st.spinner("Ingesting document..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(f"{API_URL}/ingest", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"Success! {data.get('chunk_count')} chunks created in {data.get('processing_time_sec'):.2f}s")
                        else:
                            st.error(f"Error ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")
            else:
                st.warning("Please select a file first.")
                
        st.divider()
        st.header("Settings")
        st.slider("Top-k Retrieval", min_value=1, max_value=10, value=5, key="top_k")
