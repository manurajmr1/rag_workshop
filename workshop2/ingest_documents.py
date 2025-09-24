#!/usr/bin/env python3
"""
Document Ingestion Script for RAG System

This script:
1. Scans the documents folder for PDF files
2. Processes each document (loading, splitting into chunks)
3. Creates embeddings for each chunk using Google's Gemini
4. Stores the vectors in a persistent ChromaDB

NOTE: For the student exercise, the implementations of `load_documents`
and `process_documents` are intentionally left as placeholders for you to fill.
"""

import os
import glob
import logging
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Use environment variable for API key (do NOT hard-code secrets)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_documents() -> List[Document]:
    """
    TODO: Student exercise
    Load PDF documents from the documents directory and return a list of langchain Document objects.
    YOU CAN USE CHATGPT TO HELP YOU WITH THIS!!!!!!!!!!!!

    Tasks for students to complete:
    - Check if DOCUMENTS_DIR exists; if not, log an error and return an empty list.
    - Use DirectoryLoader with document directory as DOCUMENTS_DIR, loader_cls=PyPDFLoader and glob="**/*.pdf" to load PDFs and store to a variable.
    - Return a List[Document] which should be result of the previous loading step.
    - Finally remove the line raise NotImplementedError(....) at the end of this function.

    Documentation References:
    LangChain DirectoryLoader Documentation References:
    - DirectoryLoader: https://python.langchain.com/docs/how_to/document_loader_directory/
    """
    raise NotImplementedError("Implement load_documents() to load PDFs from DOCUMENTS_DIR")

def process_documents(documents: List[Document]) -> List[Document]:
    """
    TODO: Student exercise
    Split the input documents into smaller chunks and return the chunks.
    YOU CAN USE CHATGPT TO HELP YOU WITH THIS!!!!!!!!!!!!

    Tasks for students to complete:
    - Use RecursiveCharacterTextSplitter with chunk_size=CHUNK_SIZE and chunk_overlap=CHUNK_OVERLAP.
    - Specifically use split_documents(documents) and dont use split_text() then store the chunks to a variable called chunks.
    - Return a List[Chunks] which should be the result of the splitting process from previous step.
    - Finally remove the line raise NotImplementedError(....) at the end of this function whic is just a placeholder.

    Documentation References:
    Link - https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TextSplitter.html#langchain_text_splitters.base.TextSplitter.split_documents
           https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
           https://python.langchain.com/docs/concepts/text_splitters/#why-split-documents
    """
    
    raise NotImplementedError("TODO: Implement process_documents() to split documents into chunks")

def create_vector_store(chunks: List[Document]) -> None:
    """
    Create and save vector store from document chunks
    """
    logger.info("Creating embeddings and vector store")

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set. Set the environment variable and retry.")
        return

    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Create the vector store with persistence
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # No need to call persist() - the data is automatically persisted with the persist_directory parameter
    logger.info(f"Vector store created and saved to {DB_DIR}")

    # Return some stats (if available)
    try:
        collection = vector_store._collection
        logger.info(f"Collection name: {collection.name}")
        logger.info(f"Collection count: {collection.count()}")
    except Exception:
        logger.debug("Could not fetch collection stats; this may depend on Chroma internals.")

def main():
    """Main function to run the ingestion process"""
    logger.info("Starting document ingestion process")

    # Load documents
    documents = load_documents()
    if not documents:
        logger.warning("No documents found. Please add documents to the documents directory.")
        return

    # Process documents
    chunks = process_documents(documents)

    # Create vector store
    create_vector_store(chunks)

    logger.info("Document ingestion complete!")

if __name__ == "__main__":
    main()
