#!/usr/bin/env python3
"""
RAG Query Interface

# Create a retriever from the vector store with MMR for diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for more diversity
        search_kwargs={
            "k": 5,  # Return top 5 most relevant chunks
            "fetch_k": 20,  # Consider top 20 before selecting diverse subset
            "lambda_mult": 0.5  # Diversity factor (0 = max diversity, 1 = max relevance)
        }
    ):
1. Loads the previously created vector database
2. Accepts user queries via command line
3. Retrieves relevant document chunks based on query similarity
4. Sends those chunks along with the query to a Gemini LLM
5. Returns the LLM's response that incorporates the document knowledge
"""

import os
import logging
from typing import List, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_DIR = os.path.join(os.path.dirname(__file__), "db")

# Use default API key if none is found in environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_retriever():
    """Load the vector store and create a retriever"""
    logger.info(f"Loading vector store from {DB_DIR}")
    
    if not os.path.exists(DB_DIR):
        logger.error(f"Vector store directory does not exist: {DB_DIR}")
        logger.error("Please run ingest_documents.py first to create the vector store.")
        return None
    
    # Initialize the embedding function - must match what was used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load the persisted vector store
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Return top 5 most similar chunks
    )
    
    logger.info("Vector store loaded successfully")
    collection = vector_store._collection
    logger.info(f"Collection has {collection.count()} documents")
    
    return retriever

def create_rag_chain(retriever):
    """Create the RAG chain with the retriever and LLM"""
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Using gemini-2.5-flash for better performance
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2  # Lower temperature for more factual responses
    )
    
    # Create a prompt template for the LLM
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            
            Use ONLY the information from the provided CONTEXT to answer the question. 
            If the answer cannot be found in the context, say "I don't have enough information to answer that question."
            Always cite specific information from the context by mentioning the document name if available.
            
            CONTEXT:
            {context}
            """),
            ("human", "{input}")
        ]
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Create the RAG chain that combines retrieval and generation
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    return rag_chain

def main():
    """Main function to run the query interface"""
    logger.info("Starting RAG query interface")
    
    # Load the retriever
    retriever = load_retriever()
    if not retriever:
        return
    
    # Create the RAG chain
    rag_chain = create_rag_chain(retriever)
    
    print("\n" + "=" * 50)
    print("Welcome to the Document Q&A System")
    print("Ask any question about your documents")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 50 + "\n")
    
    # Main interaction loop
    while True:
        # Get user query
        query = input("\nYour question: ")
        
        # Check for exit command
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Document Q&A System. Goodbye!")
            break
        
        try:
            # Process the query through the RAG chain
            response = rag_chain.invoke({"input": query})
            
            # Extract and print the answer
            answer = response.get('answer', "I'm sorry, I couldn't generate a response.")
            print("\nAnswer:", answer)
            
            # Optionally show source documents
            if 'context' in response and response['context']:
                print("\nRelevant document sources:")
                # Track unique sources only
                unique_sources = set()
                for doc in response['context']:
                    source = doc.metadata.get('source', 'Unknown source')
                    source_name = os.path.basename(source)
                    unique_sources.add(source_name)
                
                # Display unique sources only
                for i, source in enumerate(sorted(unique_sources), 1):
                    print(f"{i}. {source}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("Sorry, there was an error processing your request.")

if __name__ == "__main__":
    main()
