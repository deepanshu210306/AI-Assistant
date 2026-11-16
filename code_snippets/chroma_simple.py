import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")

def get_embeddings(text, model=EMBED_MODEL):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

chunks={
    "1": "Leave Policy: Employees are entitled to 20 days of paid leave per year.\nSick Leave: Employees can take up to 10 days of sick leave per year.\nMaternity Leave: Female employees are entitled to 12 weeks of maternity leave.\nPaternity Leave: Male employees are entitled to 2 weeks of paternity leave.\n",
    "2": "Medical Insurance: The company provides comprehensive medical insurance coverage for all employees.\nRetirement Benefits: Employees are eligible for a retirement benefits plan after 5 years of service.\n",
    "3": "Work From Home Policy: Employees can work from home up to 2 days per week with prior approval from their manager.\nProfessional Development: The company encourages employees to attend workshops and training programs for skill development.\n"
}

ids = list(chunks.keys())
documents = list(chunks.values())
embeddings = [get_embeddings(doc) for doc in documents]

chroma =chromadb.persistent_client(path="./chromadbStorage")

zoro_collection = chroma.get_or_create_collection(
    name="zoro_maps_pvt_ltd",
    configuration={
        'hnsw':{
            'space':'cosine',
        }

    }
)

zoro_collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)
query="What is the medical policy for employees?"
query_embedding = get_embeddings(query)
results = zoro_collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("Results from ChromaDB:")
print(results["documents"])
