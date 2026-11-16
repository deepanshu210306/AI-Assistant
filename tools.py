from datetime import datetime
import requests
import json
import os
from Utilities.utilities import log
from rag import search_documents
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    reponse = requests.get(url,timeout=10)
    reponse.raise_for_status()  # Raise an exception for HTTP errors
    return reponse.text


def run_tool(tool_call):
    try:
        func_name = tool_call.function.name
        func_args = json.loads(
            tool_call.function.arguments or "{}"
        )
        log(f"  calling function '{func_name}' with arguments: {func_args}")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        result = fn(**func_args)
    
    except Exception as e:
        result = f"Error: {str(e)}"
        log(f"   -> result: {result}")
    return result

def search_knowledge_base(query):
    results = search_documents(query,2)
    log(f"knowledge base search for: '{query}'")
    for r in results:
        log(f" {r['score']:.4f}, {r['source']}, #{r['chunk_id']}")

    blocks = []
    for r in results:
        source = r["source"]
        chunk_id = r["chunk_id"]
        chunkText = r["text"]
        header = f"[{source}, #{chunk_id}]"
        block=f"{header}\n{chunkText}"
        blocks.append(block)
    return "\n\n".join(blocks)


TOOLS=[
    {
        "type": "function",
        "function":{
            "name": "get_current_time",
            "description": "Get the current date and time in the format YYYY-MM-DD HH:MM:SS",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function":{
            "name": "get_city_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city to get the weather for."}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "search_knowledge_base",
            "description": "search the company's knowldege base for relevant information based on the user's query."
                           "Use this to answer any  que based on company's knowledge base."
                           "If the answer is not found in the knowledge base, respond with 'No relevant information found in the knowledge base."
                           "return the most relevant information from the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to find relevant information in the knowledge base."},
                },
                "required": ["query"]
            }
        }
    }
]


FUNCTION_NAME_TO_FUNCTION_MAP={
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "search_knowledge_base": search_knowledge_base
}