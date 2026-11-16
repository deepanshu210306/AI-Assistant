import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")  
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

# Generate embeddings for a given text
def get_embeddings(text, model=EMBED_MODEL):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Load documents from the knowledge base folder
def load_document(folder=KB_FOLDER):
    documents=[]

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path =os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "source": filename,
                    "text": f.read()
                }
            )
    return documents

# Add embeddings to each chunk in the list
def add_embeddings(chunks):
    for chunk in chunks:
        chunk["vector"] = get_embeddings(chunk["text"])
    return chunks

# Chunk a document into smaller parts based on double newlines
def chunk_document(document):
    chunks = []
    chunk_id=0;
    for part in document["text"].split("\n\n"):
        if part.strip():
            chunks.append({
                "source": document["source"],
                "chunk_id": len(chunks) + 1,
                "text": part.strip()
            })
            chunk_id += 1
    return chunks

# Create chunks for all documents
def create_chunks(documents):
    chunks=[]
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks

# Save and load vector store functions
def save_vector_store(chunks, file_path=VECTOR_STORE_FILE):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

# Load vector store from a JSON file
def load_vector_store(file_path=VECTOR_STORE_FILE):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks():
    if os.path.exists(VECTOR_STORE_FILE):
        print(f"Loading vector store from {VECTOR_STORE_FILE}...")
        return load_vector_store(VECTOR_STORE_FILE)
    else:
        documents = load_document(KB_FOLDER)
        chunks = create_chunks(documents)
        chunks = add_embeddings(chunks)
        save_vector_store(chunks, VECTOR_STORE_FILE)
        print(f"Vector store created and saved to {VECTOR_STORE_FILE}.")
        return chunks

chunks= load_chunks()

# Search for top_k chunks based on cosine similarity to the query
def search_documents(query, top_k=2):
    query_vector = get_embeddings(query)
    results = []

    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append({
            "score": float(score),
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "size": len(chunk["text"])
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
