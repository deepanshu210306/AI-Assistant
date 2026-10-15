from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)
messages = []
messages.append(
    {
        "role": "system",
        "content": "You are a helpful assistant and always answer in a concise manner. If you don't know the answer, say 'I don't know'."
    }
)

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        break
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages            
    )
    messages.append(
        {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
    )
    print("Assitant: " + response.choices[0].message.content)