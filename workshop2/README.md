# Real-World RAG System

This is a practical implementation of a Retrieval-Augmented Generation (RAG) system that allows you to query your own PDF documents using Google's Gemini models.

## Directory Structure

```
real_world_rag/
├── documents/     # Place your PDF files here
├── db/            # The persistent ChromaDB vector database will be stored here
├── ingest_documents.py  # Script to process documents and create vector embeddings
└── query_documents.py   # Script to query the system with natural language
```

## Setup

### Step 1: Create and Activate Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 2: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables (Optional)

The system uses a default Google API key, but you can set your own:

```bash
export GOOGLE_API_KEY=your_google_api_key_here
```

## How to Use

### Step 1: Add Your Documents

Place your PDF files in the `documents/` directory. The system supports:
- PDF files (*.pdf)

### Step 2: Ingest and Process Documents

Run the ingestion script to process all documents and create the vector database:

```bash
python ingest_documents.py
```

This script will:
1. Load all documents from the documents directory
2. Split them into manageable chunks
3. Create vector embeddings for each chunk using Google's Gemini embedding model
4. Store these vectors in a persistent ChromaDB database in the `db/` directory

### Step 3: Query Your Documents

After ingestion is complete, you can query your documents using natural language:

```bash
python query_documents.py
```

This will start an interactive session where you can ask questions about your documents. The system will:
1. Convert your query to an embedding
2. Find the most relevant document chunks
3. Send those chunks along with your query to Gemini
4. Return a response that incorporates knowledge from your documents

Type 'exit' or 'quit' to end the session.

## Customization

You can modify the following parameters in the scripts:
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- LLM model: Currently using "gemini-2.5-flash" for queries
- Number of retrieved documents: Currently set to 5 most relevant chunks

## Requirements

- Python 3.8 or higher
- Required packages: langchain, langchain-google-genai, langchain-chroma, etc.
