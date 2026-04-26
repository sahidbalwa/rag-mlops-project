"""
Streamlit chat interface component.
=== FILE: frontend/components/chat.py ===
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8002")

def render_chat():
    """
    Renders the main chat window and handles messaging.
    """
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Render sources if attached
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.caption(f"**[{i+1}] {source['source']}** (Score: {source['relevance_score']:.3f})")
                        st.write(source["content_snippet"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant thinking
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                top_k = st.session_state.get("top_k", 5)
                payload = {
                    "question": prompt,
                    "top_k": top_k
                }
                
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    
                    message_placeholder.markdown(answer)
                    
                    if sources:
                        with st.expander("Sources"):
                            for i, source in enumerate(sources):
                                st.caption(f"**[{i+1}] {source['source']}** (Score: {source['relevance_score']:.3f})")
                                st.write(source["content_snippet"])
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
            except Exception as e:
                error_msg = f"Failed to connect to API: {e}"
                message_placeholder.error(error_msg)
