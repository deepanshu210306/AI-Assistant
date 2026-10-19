from datetime import datetime
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    reponse = requests.get(url,timeout=10)
    if reponse.status_code == 200:
        return reponse.text
    else:
        return f"Could not fetch weather for {city}. Please try again later."
    


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