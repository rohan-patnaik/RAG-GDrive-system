import streamlit as st

st.set_page_config(
    page_title="RAG GDrive System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 RAG GDrive System")
st.markdown("**Retrieval-Augmented Generation with Multi-LLM Support**")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
Welcome to the RAG GDrive System! Use the pages in the sidebar to:

- **Query System**: Ask questions about your documents
- **System Status**: Check the health of all components  
- **Ingest Documents**: Add new documents to the system

---
**Backend API**: Running on http://localhost:8000
""")

# Main page content
st.markdown("""
## About This System

This RAG (Retrieval-Augmented Generation) system allows you to:

### 🔄 **Multi-Step Process**
1. **Ingest** documents into the vector database
2. **Query** using natural language
3. **Retrieve** relevant document chunks
4. **Generate** AI-powered answers with context

### 🤖 **Supported LLM Providers**
- **OpenAI** (GPT models)
- **Anthropic** (Claude models) 
- **Google Gemini**

### 📊 **Key Features**
- Document chunking and embedding
- Vector similarity search
- Multi-provider LLM integration
- Real-time system health monitoring

---
👈 **Get started by selecting a page from the sidebar!**
""")

# Display system info in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Backend Status", "🟢 Running", "Port 8000")

with col2:
    st.metric("Frontend Status", "🟢 Active", "Port 8501")
    
with col3:
    st.metric("Ready to Use", "✅ Yes", "All systems operational")
