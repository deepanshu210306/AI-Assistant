from datetime import datetime
import requests
import json
import os
import Utilities.utilities as log
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
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP={
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather
}