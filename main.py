import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import process_documents
from src.rag_engine import get_rag_chain

# Load environment variables (API Keys)
load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="DocuMind | Cloud RAG",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"  # <--- Forces sidebar to be OPEN
)

# --- 2. Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

# --- 3. Sidebar UI ---
with st.sidebar:
    # Logo Fix: Safely load logo or skip if missing
    if os.path.exists("assets/logo.png"):
        try:
            st.image("assets/logo.png", width=60)
        except Exception:
            pass 
            
    st.title("📂 Document Hub")
    st.markdown("Upload PDFs to power the cloud brain.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    # Process Button
    if st.button("🔄 Process Documents"):
        if uploaded_files:
            with st.spinner("Analyzing & Embedding documents..."):
                # Check for API Key before processing
                if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
                    st.error("❌ API Key missing! Check your .env file.")
                else:
                    success = process_documents(uploaded_files)
                    if success:
                        st.success("✅ Knowledge Base Updated!")
                    else:
                        st.error("Failed to process documents.")
        else:
            st.warning("Please upload a file first.")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("☁️ Cloud Engine: Active")
    st.caption("• Model: Zephyr 7B (HF API)")

# --- 4. Main Chat Interface ---
st.title("☁️ DocuMind Cloud")
st.markdown("#### Intelligent Document Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get the Cloud Chain
            chain = get_rag_chain()
            
            # Run the chain (Standard invoke is safer for Free Cloud API)
            full_response = chain.invoke(prompt)
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")