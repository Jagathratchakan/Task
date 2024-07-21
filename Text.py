import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama

# Function to extract text from PDF using PyMuPDF
"""
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    doc.close()
    return text """

# Function to split documents into chunks
def split_documents_into_chunks(data, chunk_size=500, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

# Function to create embeddings using Sentence Transformers
def create_embeddings(docs, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedded_docs = []
    for doc in docs:
        embedded_chunks = []
        for chunk in doc:
            embeddings = model.encode([chunk])
            embedded_chunks.append({'text': chunk, 'embeddings': embeddings[0]})
        embedded_docs.append(embedded_chunks)
    return embedded_docs

# Function to build FAISS index
def create_faiss_index(embedded_docs):
    embeddings = [chunk['embeddings'] for doc in embedded_docs for chunk in doc]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index
        
        
# Function to Vector Search using FAISS index
def index_based_search(index, text, question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([question])[0]
    D, I = index.search(query_embedding.reshape(1, -1), 5)  
    relevant_texts = [text[i] for i in I[0]]
    return relevant_texts

# Main function
if __name__ == "__main__":

    # Step 1: Extract text from PDF
    loader = PyPDFLoader("./BERT.pdf")
    data = loader.load()
    docu = split_documents_into_chunks(data)

    # Step 2: Split documents into chunks
    embedded_docs = create_embeddings(docu)

    # Step 3: Create embeddings
    index = create_faiss_index(embedded_docs)
    
    # Step 5: Initialize Sentence Transformers model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Step 6: Vector Search 
    query = "Unsupervised Feature-based Approaches "
    result = index_based_search(index,docu,query)
