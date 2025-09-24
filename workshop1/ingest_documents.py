#!/usr/bin/env python3
"""
Document Ingestion Script for RAG System

This script:
1. Scans the documents folder for PDF files
2. Processes each document (loading, splitting into chunks)
3. Creates embeddings for each chunk using Google's Gemini
4. Stores the vectors in a persistent ChromaDB
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
# from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Use default API key if none is found in environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_documents() -> List[Document]:
    """
    Load PDF documents from the documents directory
    """
    logger.info(f"Loading PDF documents from {DOCUMENTS_DIR}")
    
    # Check if directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        logger.error(f"Directory does not exist: {DOCUMENTS_DIR}")
        return []
    
    # Load PDFs
    pdf_loader = DirectoryLoader(
        DOCUMENTS_DIR, 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # Check if PDFs exist and load them
    if not glob.glob(os.path.join(DOCUMENTS_DIR, "**/*.pdf"), recursive=True):
        logger.warning("No PDF documents found in the documents directory")
        return []
    
    # Load PDF documents
    documents = pdf_loader.load()
    
    logger.info(f"Loaded {len(documents)} PDF documents")
    return documents

def process_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks
    """
    logger.info("Splitting documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    
    return chunks

def create_vector_store(chunks: List[Document]) -> None:
    """
    Create and save vector store from document chunks
    """
    logger.info("Creating embeddings and vector store")
    
    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    # client = genai.Client()

    # result = client.models.embed_content(
    #     model="gemini-embedding-001",
    #     contents="What is the meaning of life?")
    # embeddings = result.embeddings
    # Create the vector store with persistence
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    # No need to call persist() - the data is automatically persisted with the persist_directory parameter
    logger.info(f"Vector store created and saved to {DB_DIR}")
    
    # Return some stats
    collection = vector_store._collection
    logger.info(f"Collection name: {collection.name}")
    logger.info(f"Collection count: {collection.count()}")

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
