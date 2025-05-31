import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
import time

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import get_system_status

st.set_page_config(page_title="System Status", page_icon="🔧")

st.title("🔧 System Status")
st.markdown("Monitor the health and status of all RAG system components.")

# Auto-refresh toggle with proper implementation
auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)

# Manual refresh button
col1, col2 = st.columns([1, 4])
with col1:
    manual_refresh = st.button("🔄 Refresh Status", type="primary")

with col2:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh logic - use session state to track timing
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= 30:  # 30 seconds
        st.session_state.last_refresh = current_time
        st.rerun()
    
    # Show countdown
    time_since_refresh = current_time - st.session_state.last_refresh
    time_remaining = max(0, 30 - time_since_refresh)
    st.info(f"⏱️ Next auto-refresh in {time_remaining:.0f} seconds")

# Force refresh on manual button click
if manual_refresh:
    if 'last_refresh' in st.session_state:
        st.session_state.last_refresh = time.time()
    st.rerun()

# Fetch and display status
try:
    with st.spinner("Checking system status..."):
        status_response = asyncio.run(get_system_status())
    
    if status_response:
        # Overall system status
        overall_status = status_response.get('system_status', 'UNKNOWN')
        
        # Status color mapping
        status_colors = {
            'OK': '🟢',
            'DEGRADED': '🟡', 
            'ERROR': '🔴',
            'UNKNOWN': '⚪'
        }
        
        status_icon = status_colors.get(overall_status, '⚪')
        
        st.subheader(f"{status_icon} Overall Status: {overall_status}")
        
        # System info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("App Name", status_response.get('app_name', 'N/A'))
            
        with col2:
            st.metric("Environment", status_response.get('environment', 'N/A'))
            
        with col3:
            st.metric("Version", status_response.get('version', 'N/A'))
        
        # Component statuses
        st.subheader("📊 Component Status")
        
        components = status_response.get('components', [])
        if components:
            for component in components:
                name = component.get('name', 'Unknown')
                comp_status = component.get('status', 'UNKNOWN')
                message = component.get('message', 'No message')
                details = component.get('details', {})
                
                comp_icon = status_colors.get(comp_status, '⚪')
                
                with st.expander(f"{comp_icon} {name} - {comp_status}"):
                    st.markdown(f"**Status:** {comp_status}")
                    st.markdown(f"**Message:** {message}")
                    
                    if details:
                        st.markdown("**Details:**")
                        st.json(details)
        else:
            st.warning("No component information available.")
        
        # Raw response
        with st.expander("🔍 Raw Status Response"):
            st.json(status_response)
            
    else:
        st.error("❌ Could not retrieve system status.")
        
except Exception as e:
    st.error(f"❌ Error checking system status: {str(e)}")
    
    with st.expander("🐛 Error Details"):
        st.exception(e)

# System requirements info
st.markdown("---")
st.subheader("ℹ️ System Requirements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Backend Services:**
    - FastAPI server (Port 8000)
    - ChromaDB vector store
    - Embedding service (Sentence Transformers)
    - LLM clients (OpenAI, Anthropic, Gemini)
    """)

with col2:
    st.markdown("""
    **Health Check Endpoints:**
    - `/health` - Overall system health
    - `/query/status` - Query system status
    - `/docs` - API documentation
    """)
