# PDF RAG Chat

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

## 📌 Goal

This project implements a **Retrieval-Augmented Generation (RAG)** system for querying a large amount of PDF documents using a local Ollama server with Open-Source models, LangChain and a Streamlit-based UI. By combining vector embeddings, a Chroma vector store, and LLMs, it delivers accurate, context-aware answers to user queries over uploaded PDF data.

![Image](https://github.com/user-attachments/assets/beb5afaa-8f26-49a6-8e6c-4087252f4226)

## 🖼️ Demonstration

Here’s a concise demonstration using three years of Spotify’s annual reports (over 600 pages of dense content). By employing only the basic nomic-embed-text embedding model alongside a simple open-source Mistral LLM (4b parameters) and without any parameter tuning, this RAG system consistently retrieves accurate, contextually grounded information from a very large document collection.

![Image](https://github.com/user-attachments/assets/ee739cbc-96a0-45ea-9cee-694487614226)

## 🛠️ How It Works

1. **Data Ingestion & Preprocessing**

   * PDFs in the `data/` directory are loaded via `PyPDFDirectoryLoader`.
   * Documents are split into chunks using `RecursiveCharacterTextSplitter` with customizable `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters (cf. config.py or the streamlit sidebar).

2. **Vector Embedding & Storage**

   * Each chunk is embedded using `OllamaEmbeddings` (configurable via `EMBEDDING_MODEL`).
   * Embedded vectors are persisted in a ChromaDB instance at `CHROMA_PATH`.
   * Duplicate chunks are detected by unique IDs and only new chunks are added. This enables the user to update his current set of PDFs easily.

3. **Retrieval & Generation**

   * For a given query, the top *k* candidate chunks (configurable via `NUMBER_OF_CANDIDATES`) are retrieved by similarity search.
   * Retrieved contexts are assembled into a defined prompt template.
   * The prompt is sent to an Ollama LLM (`OLLAMA_RESPONSE_MODEL`) to generate a grounded response with source citations.

4. **Streamlit UI**

   * A simple interface (`ui.py`) allows:

     * Uploading PDF files
     * Configuring model & database parameters via sidebar
     * Populating / resetting the vector database
     * Entering queries and viewing answers in real-time

5. **Testing**

   * A suite in `test_rag.py` validates RAG outputs against expected responses using an Ollama hosted LLM.

## 🚀 Installation

### Prerequisites

* [Ollama](https://ollama.com/): Install and configure a local Ollama server:

  ```bash
  # Install Ollama (Mac/Linux/Windows WSL)
  curl -fsSL https://ollama.com/install.sh | sh
    ```
  # Pull default models
  ```bash
  ollama pull nomic-embed-text
  ollama pull mistral
  ```
  # Start Ollama server in a seperate Terminal
  ```bash
  ollama serve
  ```


### Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Command-Line Interface

1. **Populate Database** (add new chunks)

   ```bash
   python populate_database.py
   ```

2. **Reset Database** (clear & rebuild)

   ```bash
   python populate_database.py --reset
   ```

3. **Query via CLI**

   ```bash
   python query_data.py "Your question here"
   ```

### Streamlit UI

1. Launch the Web UI:

   ```bash
   streamlit run ui.py
   ```
2. Open your browser at `http://localhost:8501`.
3. (Optional) Use the sidebar to adjust settings:

   * Embedding & response model names (use the exact names of the models you've installed)
   * Chunk size & overlap
   * Number of candidate chunks

*(remember to rerun the UI page when editing the sidebar values)*
4. Upload PDFs, populate/reset the database, and ask questions interactively.

## ⚙️ Configuration

| Key                     | Description                                | Default            |
| ----------------------- | ------------------------------------------ | ------------------ |
| `CHROMA_PATH`           | Directory for ChromaDB persistence         | `chroma/`          |
| `DATA_PATH`             | Directory for source PDF files             | `data/`            |
| `EMBEDDING_MODEL`       | Ollama embedding model name                | `nomic-embed-text` |
| `OLLAMA_RESPONSE_MODEL` | Ollama LLM model for answer generation     | `mistral`          |
| `CHUNK_SIZE`            | Max tokens per document chunk              | `700`              |
| `CHUNK_OVERLAP`         | Overlap tokens between chunks              | `80`               |
| `NUMBER_OF_CANDIDATES`  | Number of top chunks to retrieve per query | `9`                |

*You can edit these values directly in `config.py` or via the Streamlit sidebar. (remember to rerun the UI page when editing the sidebar values)*

## 📂 Project Structure

```text
.
├── config.py                 # Core configuration constants
├── get_embedding_function.py # Returns OllamaEmbeddings instance
├── populate_database.py      # CLI entrypoint for database operations
├── populate_database_callable.py # Callable version for UI integration
├── query_data.py             # CLI query & generation logic
├── ui.py                     # Streamlit application
├── test_rag.py               # Automated tests (pytest)
├── requirements.txt          # Python dependencies
└── data/                     # Place your PDF files here (as an example, I have added the annual reports of Spotify - over 600 pages of dense content - to show how useful RAG can be for corporate productivity)
    └── *.pdf
```

## 🔍 Testing

Run this pytest suite in order to validate RAG outputs against expected responses using an Ollama hosted LLM (choose in test_rag.py). 
*By default, I used Mistral on the same local Ollama server as the answer reconstruction model.*

```bash
pytest
```
---