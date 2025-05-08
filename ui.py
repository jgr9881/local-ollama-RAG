import os
import subprocess
import importlib
import streamlit as st
from pathlib import Path
import sys
import shutil
import gc

import config

from get_embedding_function import get_embedding_function
from query_data import query_rag
from populate_database_callable import main as populate_main

st.set_page_config(page_title="RAG PDF Chat", layout="wide")

# Sidebar: Configuration
st.sidebar.header("Settings")
# Editable parameters
emb_model = st.sidebar.text_input(
    "Embedding Model", value=config.EMBEDDING_MODEL
)
resp_model = st.sidebar.text_input(
    "Response Model", value=config.OLLAMA_RESPONSE_MODEL
)
chunk_size = st.sidebar.number_input(
    "Chunk Size", min_value=100, max_value=5000, step=50, value=config.CHUNK_SIZE
)
chunk_overlap = st.sidebar.number_input(
    "Chunk Overlap", min_value=0, max_value=chunk_size - 1, step=10, value=config.CHUNK_OVERLAP
)
number_of_candidates = st.sidebar.number_input(
    "Number of Candidate Chunks", min_value=0, step = 1, value=config.NUMBER_OF_CANDIDATES
)

if st.sidebar.button("Save Config"):
    # overwrite config
    cfg_path = Path(__file__).parent / "config.py"
    content = f"""CHROMA_PATH = \"{config.CHROMA_PATH}\"
DATA_PATH = \"{config.DATA_PATH}\"

EMBEDDING_MODEL = \"{emb_model}\"
OLLAMA_RESPONSE_MODEL = \"{resp_model}\"

CHUNK_SIZE = {chunk_size}
CHUNK_OVERLAP = {chunk_overlap}
NUMBER_OF_CANDIDATES = {number_of_candidates}
"""
    cfg_path.write_text(content)
    importlib.reload(config)
    st.experimental_rerun()

st.title("📄 PDF RAG Chat")

# Section: PDF Uploader & File List
st.header("1. Manage PDF Data")
uploaded = st.file_uploader(
    "Drag & drop PDF files to upload to the data folder (or browse)",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded:
    os.makedirs(config.DATA_PATH, exist_ok=True)
    for file in uploaded:
        dest = Path(config.DATA_PATH) / file.name
        with open(dest, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Saved {file.name}")

# List current PDFs
st.subheader("Current PDFs in data folder:")
pdfs = list(Path(config.DATA_PATH).glob("*.pdf"))
if pdfs:
    for p in pdfs:
        st.write(f"- {p.name}")
else:
    st.write("_(no PDFs found)_")

# Section: Populate / Reset Database
st.header("2. Database Operations")
col1, col2 = st.columns(2)
with col1:
    if st.button("Populate Database"):
        with st.spinner("Updating database…"):
            try:
                populate_main(reset=False)
                st.success("Database populated successfully")
            except Exception as e:
                st.error(f"Failed to populate: {e}")


with col2:
    if st.button("Reset Database"):
        # file-handles close
        st.cache_resource.clear()

        with st.spinner("Clearing the database…"):
            try:
                populate_main(reset=True)
                st.success("Database reset successfully")
                # rerun so the freshly-created client is re-cached
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to reset: {e}")

# Section: Chat Interface
st.header("3. Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Ask"):
    if not question.strip():
        st.error("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = query_rag(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during query: {e}")