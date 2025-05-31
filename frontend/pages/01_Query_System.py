import streamlit as st
import asyncio
import json
from typing import Dict, Any
import sys
import os

# Add the frontend directory to the path so we can import api_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import query_backend

st.set_page_config(page_title="Query System", page_icon="❓")

st.title("❓ Query the RAG System")
st.markdown("Enter your question below to query the documents ingested into the RAG system. The system will retrieve relevant information and generate an answer using an LLM.")

# Initialize session state for query text
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Example queries section (moved to top for better UX)
st.subheader("💡 Example Queries")
st.markdown("Click on any example below to use it as your query:")

example_queries = [
    "What is Artificial Intelligence?",
    "How does machine learning work?", 
    "Tell me about Python for data science",
    "What are the key features of RAG systems?",
    "Explain deep learning and neural networks"
]

# Create example query buttons in a grid
cols = st.columns(3)
for i, example in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
            st.session_state.query_text = example
            st.rerun()

st.markdown("---")

# Query input - use session state value
query_text = st.text_area(
    "Enter your question here:", 
    value=st.session_state.query_text,
    height=100, 
    placeholder="e.g., What is Artificial Intelligence? How does machine learning work?",
    key="query_input"
)

# Update session state when text changes
if query_text != st.session_state.query_text:
    st.session_state.query_text = query_text

# Advanced options in an expander
with st.expander("🔧 Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        llm_provider = st.selectbox(
            "LLM Provider:",
            options=["gemini", "openai", "anthropic"],
            index=0,
            help="Choose which LLM provider to use for generating the answer"
        )
        
        top_k = st.slider(
            "Number of chunks to retrieve:",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve for context"
        )
    
    with col2:
        llm_model = st.text_input(
            "Specific Model (optional):",
            placeholder="e.g., gpt-4, claude-3-opus-20240229",
            help="Leave empty to use the provider's default model"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for retrieved chunks"
        )

# Clear query button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🗑️ Clear", help="Clear the query text"):
        st.session_state.query_text = ""
        st.rerun()

# Query button
query_button_disabled = not st.session_state.query_text.strip()
if st.button("🔍 Get Answer", type="primary", disabled=query_button_disabled, use_container_width=True):
    if st.session_state.query_text.strip():
        
        # Show what's happening
        status_container = st.container()
        progress_container = st.container()
        
        with status_container:
            st.info("🚀 **Processing your query...**")
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        try:
            # Step 1: Preparing request
            status_text.text("📝 Preparing query request...")
            progress_bar.progress(20)
            
            # Prepare query data
            query_data = {
                "query_text": st.session_state.query_text.strip(),
                "llm_provider": llm_provider,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
            
            # Add model if specified
            if llm_model.strip():
                query_data["llm_model_name"] = llm_model.strip()
            
            # Step 2: Sending to backend
            status_text.text("🌐 Sending request to backend...")
            progress_bar.progress(40)
            
            # Make the API call
            response = asyncio.run(query_backend(query_data))
            
            # Step 3: Processing response
            status_text.text("🤖 Processing AI response...")
            progress_bar.progress(80)
            
            # Step 4: Complete
            progress_bar.progress(100)
            status_text.text("✅ Query completed successfully!")
            
            # Clear the progress indicators after a short delay
            import time
            time.sleep(1)
            status_container.empty()
            progress_container.empty()
            
            if response:
                st.success("✅ **Query completed successfully!**")
                
                # Display the answer
                st.subheader("🤖 Generated Answer")
                
                # Provider and model info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Provider", response.get('llm_provider_used', 'N/A'))
                with col2:
                    st.metric("Model", response.get('llm_model_used', 'N/A'))
                
                # The actual answer
                st.markdown("### Answer:")
                answer_text = response.get('llm_answer', 'No answer generated.')
                st.markdown(f"> {answer_text}")
                
                # Display retrieved chunks
                retrieved_chunks = response.get('retrieved_chunks', [])
                if retrieved_chunks:
                    st.subheader(f"📄 Retrieved Context ({len(retrieved_chunks)} chunks)")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_score = sum(chunk.get('score', 0) for chunk in retrieved_chunks) / len(retrieved_chunks)
                        st.metric("Avg. Similarity", f"{avg_score:.3f}")
                    with col2:
                        st.metric("Chunks Retrieved", len(retrieved_chunks))
                    with col3:
                        sources = set(chunk.get('metadata', {}).get('filename', 'Unknown') for chunk in retrieved_chunks)
                        st.metric("Unique Sources", len(sources))
                    
                    # Show each chunk
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        score = chunk.get('score', 0)
                        filename = chunk.get('metadata', {}).get('filename', 'Unknown')
                        
                        # Color code based on score
                        if score >= 0.8:
                            score_color = "🟢"
                        elif score >= 0.6:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"
                        
                        with st.expander(f"{score_color} Chunk {i} - {filename} (Score: {score:.3f})"):
                            st.markdown("**Content:**")
                            st.text_area("", chunk.get('content', 'No content'), height=100, disabled=True, key=f"chunk_content_{i}")
                            
                            # Show metadata
                            if chunk.get('metadata'):
                                with st.expander("📋 Metadata"):
                                    st.json(chunk['metadata'])
                else:
                    st.warning("⚠️ No relevant chunks were retrieved for your query. Try:")
                    st.markdown("- Lowering the similarity threshold")
                    st.markdown("- Using different keywords")
                    st.markdown("- Checking if documents are properly ingested")
                
                # Raw response in expander
                with st.expander("🔍 Raw API Response (for debugging)"):
                    st.json(response)
                    
            else:
                status_container.empty()
                progress_container.empty()
                st.error("❌ **No response received from the backend.**")
                
        except ConnectionError as e:
            status_container.empty()
            progress_container.empty()
            st.error("❌ **Connection Error**")
            st.markdown("**The backend API is not reachable. Please check:**")
            st.markdown("- Is the backend running on http://localhost:8000?")
            st.markdown("- Check your network connection")
            st.markdown("- Verify the API server is properly started")
            
            with st.expander("🐛 Technical Details"):
                st.text(str(e))
                
        except Exception as e:
            status_container.empty()
            progress_container.empty()
            st.error("❌ **Error processing your query**")
            st.markdown("**An unexpected error occurred:**")
            st.markdown(f"```\n{str(e)}\n```")
            
            # Show error details in expander
            with st.expander("🐛 Full Error Details"):
                st.exception(e)

# Help section
st.markdown("---")
st.subheader("ℹ️ How to Use")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Quick Start:**
    1. Click an example query above, or type your own
    2. Optionally adjust advanced settings
    3. Click 'Get Answer' to query the system
    4. View the AI-generated response and source chunks
    """)

with col2:
    st.markdown("""
    **Tips:**
    - Use natural language questions
    - Be specific for better results
    - Check retrieved chunks to verify context
    - Adjust similarity threshold if needed
    """)

# Show current query state for debugging (can be removed in production)
if st.checkbox("🔧 Show Debug Info", value=False):
    st.json({
        "session_query_text": st.session_state.query_text,
        "current_query_text": query_text,
        "query_length": len(st.session_state.query_text),
        "button_enabled": not query_button_disabled
    })
