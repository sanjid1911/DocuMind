# -------------------------------------------------------------------------
# 1. FIX: SQLite Hack for Streamlit Cloud (MUST BE AT THE VERY TOP)
# -------------------------------------------------------------------------
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If we are running locally and don't have pysqlite3, that's fine.
    pass

# -------------------------------------------------------------------------
# 2. Imports
# -------------------------------------------------------------------------
import streamlit as st
import os
import tempfile

# -------------------------------------------------------------------------
# 3. FIX: Load API Key from Streamlit Secrets (Cloud) or .env (Local)
# -------------------------------------------------------------------------
# Try loading from .env first (for local dev)
from dotenv import load_dotenv
load_dotenv()

# If on Cloud, overwrite with the Secret Key
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# -------------------------------------------------------------------------
# 4. Import Internal Modules (After setting env vars)
# -------------------------------------------------------------------------
from src.ingestion import process_documents
from src.rag_engine import get_rag_chain

# -------------------------------------------------------------------------
# 5. The App UI
# -------------------------------------------------------------------------
st.set_page_config(page_title="DocuMind", page_icon="🧠")

st.title("🧠 DocuMind: Chat with your PDF")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Upload
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files to temp directory
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Run Ingestion
                    process_documents(file_paths)
                    st.success("✅ Documents processed! You can now chat.")
                
                except Exception as e:
                    # ✅ IMPROVED ERROR MESSAGE: Show the real error
                    st.error(f"❌ An error occurred during processing: {e}")
                    # Print detailed traceback for debugging
                    import traceback
                    st.text(traceback.format_exc())

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        try:
            chain = get_rag_chain()
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating response: {e}")