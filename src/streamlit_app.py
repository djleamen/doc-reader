"""
Streamlit web interface for the RAG Document Q&A system.
"""
import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide default Streamlit styling */
    .stApp > div {
        background-color: transparent;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
    }
    
    /* Remove white background boxes */
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .query-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        color: #333;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Source document styling */
    .source-doc {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(5px);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffd700;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-score {
        color: #ffd700;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Text styling */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Footer styling */
    .footer-style {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    
    /* Success/Error messages styling */
    .stSuccess {
        background: rgba(40, 167, 69, 0.2);
        border: 1px solid rgba(40, 167, 69, 0.4);
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.2);
        border: 1px solid rgba(220, 53, 69, 0.4);
        border-radius: 8px;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid rgba(255, 193, 7, 0.4);
        border-radius: 8px;
    }
    
    .stInfo {
        background: rgba(23, 162, 184, 0.2);
        border: 1px solid rgba(23, 162, 184, 0.4);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://rag-api:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_documents(files, index_name="default"):
    """Upload documents to the API."""
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type)))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload-documents",
            files=files_data,
            params={"index_name": index_name},
            timeout=300  # 5 minutes timeout for large files
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error uploading documents: {e}")
        return None

def query_documents(question, k=None, include_sources=True, include_scores=True, conversational=False):
    """Query the documents."""
    endpoint = "/conversational-query" if conversational else "/query"
    
    payload = {
        "question": question,
        "k": k,
        "include_sources": include_sources,
        "include_scores": include_scores
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return None

def get_index_stats():
    """Get index statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/index-stats", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def clear_conversation():
    """Clear conversation history."""
    try:
        response = requests.delete(f"{API_BASE_URL}/conversation", timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Q&A System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the API server first.")
        st.code("python -m src.api", language="bash")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Index selection
        index_name = st.text_input("Index Name", value="default", help="Name of the document index")
        
        # Query settings
        st.subheader("Query Settings")
        max_results = st.slider("Max Results", min_value=1, max_value=20, value=5, 
                               help="Maximum number of relevant documents to retrieve")
        include_sources = st.checkbox("Include Sources", value=True, 
                                    help="Include source documents in the response")
        include_scores = st.checkbox("Include Confidence Scores", value=True,
                                   help="Include similarity scores for retrieved documents")
        
        conversational_mode = st.checkbox("Conversational Mode", value=False,
                                        help="Enable conversation memory")
        
        # Index stats
        st.subheader("📊 Index Statistics")
        if st.button("Refresh Stats"):
            stats = get_index_stats()
            if stats:
                st.json(stats["stats"])
            else:
                st.warning("No index statistics available")
        
        # Clear conversation
        if conversational_mode:
            if st.button("Clear Conversation"):
                if clear_conversation():
                    st.success("Conversation cleared!")
                else:
                    st.error("Failed to clear conversation")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Document upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md'],
            help="Upload PDF, DOCX, TXT, or Markdown files"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    result = upload_documents(uploaded_files, index_name)
                    
                    if result:
                        st.success(f"✅ {result['message']}")
                        
                        if result['files_processed']:
                            st.write("**Processed files:**")
                            for file in result['files_processed']:
                                st.write(f"- {file}")
                        
                        if result['total_chunks_added'] > 0:
                            st.info(f"Added {result['total_chunks_added']} document chunks to the index")
                        
                        if result['errors']:
                            st.warning("**Errors encountered:**")
                            for error in result['errors']:
                                st.write(f"- {error}")
                    else:
                        st.error("Failed to upload documents")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Query section
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.subheader("❓ Ask Questions")
        
        # Sample questions
        sample_questions = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key findings?",
            "What methodology was used in the research?",
            "What are the conclusions and recommendations?"
        ]
        
        with st.expander("💡 Sample Questions"):
            for i, question in enumerate(sample_questions):
                if st.button(f"📝 {question}", key=f"sample_{i}"):
                    st.session_state.query_input = question
        
        # Query input
        question = st.text_area(
            "Enter your question:",
            value=st.session_state.get('query_input', ''),
            height=100,
            help="Ask any question about your uploaded documents"
        )
        
        if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
            with st.spinner("Searching for answers..."):
                result = query_documents(
                    question=question,
                    k=max_results,
                    include_sources=include_sources,
                    include_scores=include_scores,
                    conversational=conversational_mode
                )
                
                if result:
                    # Display answer
                    st.subheader("💬 Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    with st.expander("📈 Query Details"):
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.metric("Documents Retrieved", result['metadata'].get('retrieval_count', 0))
                            st.metric("Context Length", result['metadata'].get('context_length', 0))
                        with col_meta2:
                            st.metric("Model Used", result['metadata'].get('model_used', 'N/A'))
                            st.metric("Prompt Length", result['metadata'].get('prompt_length', 0))
                    
                    # Display source documents
                    if include_sources and result.get('source_documents'):
                        st.subheader("📚 Source Documents")
                        
                        for i, (doc, score) in enumerate(zip(
                            result['source_documents'], 
                            result.get('confidence_scores', [1.0] * len(result['source_documents']))
                        )):
                            with st.expander(f"📄 Source {i+1} - {doc['metadata'].get('filename', 'Unknown')}"):
                                
                                if include_scores:
                                    st.markdown(f'<span class="confidence-score">Confidence Score: {score:.4f}</span>', 
                                              unsafe_allow_html=True)
                                
                                st.markdown('<div class="source-doc">', unsafe_allow_html=True)
                                st.write(doc['content'])
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Metadata
                                if doc['metadata']:
                                    st.write("**Metadata:**")
                                    metadata_cols = st.columns(3)
                                    metadata_items = list(doc['metadata'].items())
                                    for j, (key, value) in enumerate(metadata_items):
                                        with metadata_cols[j % 3]:
                                            st.write(f"**{key}:** {value}")
                else:
                    st.error("Failed to get answer")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Conversation history in conversational mode
    if conversational_mode:
        st.subheader("💭 Conversation History")
        st.info("In conversational mode, the system remembers previous questions and answers to provide better context.")
        
        # This would need to be implemented to show actual conversation history
        # For now, just show a placeholder
        if st.session_state.get('conversation_started'):
            st.write("Conversation history would be displayed here...")
        else:
            st.write("No conversation history yet. Ask your first question!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-style">
        <p style="color: #ffffff; font-size: 1.1rem; margin: 0;">
            🤖 <strong>Powered by RAG (Retrieval-Augmented Generation)</strong> • Built with Streamlit & FastAPI
        </p>
        <p style="color: #e0e0e0; margin: 0.5rem 0 0 0;">
            Upload your documents and start asking questions!
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
