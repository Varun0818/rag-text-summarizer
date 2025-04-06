import os
import faiss
import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Configuration and Model Loading
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_NAME, device=0 if DEVICE == 'cuda' else -1)

# Chunking Function
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Embedding + FAISS Index Creation
def embed_chunks(chunks, embedder):
    return embedder.encode(chunks, convert_to_tensor=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.cpu().numpy())
    return index

# Retrieve Relevant Chunks
def retrieve_top_k_chunks(index, embeddings, chunks, top_k=TOP_K):
    query_embedding = embeddings.mean(dim=0, keepdim=True)
    _, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [chunks[i] for i in indices[0]]

# Summarize
# Constants for summary length
MAX_SUMMARY_LENGTH = 512  # Maximum tokens for summary
MIN_SUMMARY_LENGTH = 150  # Minimum tokens for summary

def summarize_chunks(chunks, summarizer, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH):
    text = "\n".join(chunks)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def summarize_text(text: str, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH):
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, embedder)
    index = build_faiss_index(embeddings)
    top_chunks = retrieve_top_k_chunks(index, embeddings, chunks)
    summary = summarize_chunks(top_chunks, summarizer, max_length=max_length, min_length=min_length)
    return summary

# Full Summarization Pipeline Function
def summarize_text(text: str):
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, embedder)
    index = build_faiss_index(embeddings)
    top_chunks = retrieve_top_k_chunks(index, embeddings, chunks)
    summary = summarize_chunks(top_chunks, summarizer)
    return summary

# Example usage
if __name__ == "__main__":
    # Read from text.txt file
    file_path = "text.txt"  # Using relative path for better portability
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Reading text from file: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please ensure 'text.txt' exists in the project directory.")
        exit(1)
    
    # Print text length to verify full content is being processed
    print(f"Processing text of length: {len(text)} characters")
    
    # Process the text
    summary = summarize_text(text)
    
    # Print the summary
    print("\nSUMMARY:")
    print(summary)