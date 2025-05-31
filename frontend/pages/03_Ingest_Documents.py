import streamlit as st
import asyncio
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import ingest_documents_on_backend

st.set_page_config(page_title="Ingest Documents", page_icon="📁")

st.title("📁 Ingest Documents")
st.markdown("Add documents to the RAG system by specifying a server-side directory path.")

# Information about the ingestion process
st.info("""
**How Document Ingestion Works:**
1. 📂 Documents are loaded from the specified server directory
2. 🧹 Text is cleaned and split into chunks
3. 🔢 Embeddings are generated for each chunk
4. 💾 Chunks are stored in the vector database (ChromaDB)
""")

# Input form
with st.form("ingestion_form"):
    st.subheader("📂 Directory Settings")
    
    server_path = st.text_input(
        "Server Directory Path:",
        value="data/sample_documents",
        help="Path to the directory containing documents (relative to the backend server)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_patterns_str = st.text_input(
            "File Patterns (comma-separated):",
            value="*.txt,*.md",
            help="File patterns to match, e.g., *.txt,*.md,*.pdf"
        )
    
    with col2:
        recursive_ingest = st.checkbox(
            "Ingest recursively (search subdirectories)", 
            value=True
        )
    
    # Submit button
    submitted = st.form_submit_button("🚀 Start Ingestion", type="primary", use_container_width=True)
    
    if submitted:
        if not server_path.strip():
            st.error("❌ Please provide a server directory path.")
        else:
            # Progress tracking containers
            status_container = st.container()
            progress_container = st.container()
            
            with status_container:
                st.info("📤 **Starting document ingestion...**")
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            try:
                # Step 1: Preparing request
                status_text.text("📝 Preparing ingestion request...")
                progress_bar.progress(10)
                
                # Prepare ingestion data
                ingestion_data = {
                    "source_directory": server_path.strip(),
                    "recursive": recursive_ingest
                }
                
                # Add file patterns if specified
                if file_patterns_str.strip():
                    patterns = [p.strip() for p in file_patterns_str.split(',') if p.strip()]
                    if patterns:
                        ingestion_data["file_patterns"] = patterns
                
                # Step 2: Sending request
                status_text.text("🌐 Sending request to backend...")
                progress_bar.progress(25)
                
                # Step 3: Processing
                status_text.text("🔄 Backend is processing documents...")
                progress_bar.progress(50)
                
                # Make the API call
                response = asyncio.run(ingest_documents_on_backend(ingestion_data))
                
                # Step 4: Finalizing
                status_text.text("✅ Ingestion completed!")
                progress_bar.progress(100)
                
                # Clear progress after short delay
                import time
                time.sleep(1)
                status_container.empty()
                progress_container.empty()
                
                if response:
                    st.success("✅ **Ingestion completed successfully!**")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        docs_processed = response.get('documents_processed', 0)
                        st.metric(
                            "Documents Processed", 
                            docs_processed,
                            delta=f"+{docs_processed}" if docs_processed > 0 else None
                        )
                    
                    with col2:
                        chunks_added = response.get('chunks_added', 0)
                        st.metric(
                            "Chunks Added", 
                            chunks_added,
                            delta=f"+{chunks_added}" if chunks_added > 0 else None
                        )
                        
                    with col3:
                        error_count = len(response.get('errors', []))
                        st.metric(
                            "Errors", 
                            error_count,
                            delta=f"{error_count} errors" if error_count > 0 else "No errors"
                        )
                    
                    # Show message
                    message = response.get('message', 'No message')
                    st.markdown(f"**Result:** {message}")
                    
                    # Show errors if any
                    errors = response.get('errors', [])
                    if errors:
                        st.warning(f"⚠️ **{len(errors)} error(s) occurred during ingestion:**")
                        for i, error in enumerate(errors, 1):
                            st.text(f"{i}. {error}")
                    
                    # Success tips
                    if docs_processed > 0 and chunks_added > 0:
                        st.info("💡 **Next step:** Go to the Query System page to ask questions about your documents!")
                    
                    # Raw response
                    with st.expander("🔍 Raw Ingestion Response"):
                        st.json(response)
                        
                else:
                    status_container.empty()
                    progress_container.empty()
                    st.error("❌ **No response received from the backend.**")
                    
            except ConnectionError as e:
                status_container.empty()
                progress_container.empty()
                st.error("❌ **Connection Error**")
                st.markdown("**Cannot connect to the backend API:**")
                st.markdown("- Is the backend running on http://localhost:8000?")
                st.markdown("- Check your network connection")
                
            except Exception as e:
                status_container.empty()
                progress_container.empty()
                st.error("❌ **Error during ingestion**")
                st.markdown(f"**Error details:** {str(e)}")
                
                with st.expander("🐛 Technical Details"):
                    st.exception(e)

# Sample documents section
st.markdown("---")
st.subheader("📄 Available Sample Documents")

sample_info = [
    {
        "file": "sample_doc_1.txt",
        "topic": "Artificial Intelligence Evolution",
        "description": "Covers AI history, deep learning, LLMs, and RAG systems"
    },
    {
        "file": "sample_doc_2.txt", 
        "topic": "Python for Data Science",
        "description": "Python libraries, data science workflow, and tools"
    }
]

for doc in sample_info:
    with st.expander(f"📄 {doc['file']} - {doc['topic']}"):
        st.markdown(f"**Topic:** {doc['topic']}")
        st.markdown(f"**Description:** {doc['description']}")

# Tips section
st.markdown("---")
st.subheader("💡 Tips for Document Ingestion")

tips = [
    "**Supported formats:** Currently supports .txt and .md files",
    "**Directory structure:** Use recursive search to include subdirectories", 
    "**File patterns:** Use wildcards like *.txt or *.pdf to filter files",
    "**Performance:** Larger documents will be automatically chunked for processing",
    "**Duplicates:** Re-ingesting the same directory will update existing chunks (see warnings in backend logs)"
]

for tip in tips:
    st.markdown(f"• {tip}")
