from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)
while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        break
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {
                "role": "user",
                "content": user_input
            },
        ]
    )

    print("Assitant:" + response.choices[0].message.content)