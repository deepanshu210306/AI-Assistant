import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")  


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

# Add embeddings to each document in the list
def add_embeddings(documents):
    for doc in documents:
        doc["vector"] = get_embeddings(doc["text"])
    return documents

# Search for top_k documents based on cosine similarity to the query
def search_documents(query, documents, top_k=3):
    query_vector = get_embeddings(query)
    results = []

    for doc in documents:
        score = cosine_similarity(query_vector, doc["vector"])
        results.append({
            "score": float(score),
            "source": doc["source"],
            "text": doc["text"],
            "size": len(doc["text"])
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def test(user_query,documents,top_k=3):
    print(f"\nUser Query: {user_query}\n")

    results = search_documents(user_query, documents, top_k)

    for r in results:
        print(f"Score: {r['score']:.4f}, Source: {r['source']}, Size: {r['size']} characters")
        # print(f"Text: {r['text']}\n")


documents = load_document(KB_FOLDER)
print(documents[0])
documents = add_embeddings(documents)
print("Documents loaded and embeddings added.")

user_query = "What is the medical policy for employees?"

test(user_query, documents, top_k=3)